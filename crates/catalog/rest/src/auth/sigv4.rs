// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`AuthManager`] that signs requests using the AWS SigV4 signing process,
//! e.g. for the AWS Glue Iceberg REST endpoint.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use aws_config::sts::AssumeRoleProvider;
use aws_config::{BehaviorVersion, Region, SdkConfig};
use aws_credential_types::Credentials;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use http::{HeaderName, HeaderValue};
use iceberg::{Error, ErrorKind, Result};
use tokio::sync::OnceCell;
use uuid::Uuid;

use super::{AuthManager, AuthSession, HttpRequest};
use crate::catalog::{PATH_V1, REST_CATALOG_PROP_URI, RestCatalogConfig};
use crate::client::HttpClient;

/// `rest.auth.type` value selecting SigV4 signing.
pub const AUTH_TYPE_SIGV4: &str = "sigv4";

const DEFAULT_SIGNING_NAME: &str = "glue";
const DEFAULT_SIGNING_REGION: &str = "us-east-1";
/// An `Authorization` header set by the delegate session is relocated here
/// before signing, as Iceberg Java's `RESTSigV4AuthSession` does.
const RELOCATED_AUTHORIZATION: &str = "original-authorization";

/// Wraps a delegate [`AuthManager`] and signs every request its sessions
/// authenticate with AWS SigV4.
///
/// Enabled by `rest.sigv4-enabled=true` (or `rest.auth.type=sigv4`) and
/// configured with `rest.signing-region`, `rest.signing-name`,
/// `rest.access-key-id`, `rest.secret-access-key`, `rest.session-token`,
/// `rest.client.assume-role.arn` and `rest.client.assume-role.session-name`.
/// Without explicit credentials the default AWS credentials chain is used.
pub(crate) struct SigV4AuthManager {
    delegate: Arc<dyn AuthManager>,
    signer: Arc<SigV4Signer>,
}

impl SigV4AuthManager {
    pub(crate) fn new(delegate: Arc<dyn AuthManager>, cfg: &RestCatalogConfig) -> Self {
        Self {
            delegate,
            signer: Arc::new(SigV4Signer::from_config(cfg)),
        }
    }

    /// Requests outside the catalog's API (e.g. to a token endpoint on another
    /// host) are left unsigned.
    fn session(
        &self,
        delegate: Arc<dyn AuthSession>,
        props: &HashMap<String, String>,
    ) -> SigV4Session {
        let catalog_prefix = props
            .get(REST_CATALOG_PROP_URI)
            .map(|uri| [uri.as_str(), PATH_V1].join("/"));
        SigV4Session {
            delegate,
            signer: self.signer.clone(),
            catalog_prefix,
        }
    }
}

impl Debug for SigV4AuthManager {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SigV4AuthManager")
            .field("delegate", &self.delegate)
            .field("signer", &self.signer)
            .finish()
    }
}

#[async_trait]
impl AuthManager for SigV4AuthManager {
    async fn init_session(
        &self,
        client: &HttpClient,
        props: &HashMap<String, String>,
    ) -> Result<Box<dyn AuthSession>> {
        let delegate = self.delegate.init_session(client, props).await?;
        Ok(Box::new(self.session(Arc::from(delegate), props)))
    }

    async fn catalog_session(
        &self,
        client: &HttpClient,
        props: &HashMap<String, String>,
    ) -> Result<Arc<dyn AuthSession>> {
        let delegate = self.delegate.catalog_session(client, props).await?;
        Ok(Arc::new(self.session(delegate, props)))
    }
}

#[derive(Debug)]
struct SigV4Session {
    delegate: Arc<dyn AuthSession>,
    signer: Arc<SigV4Signer>,
    catalog_prefix: Option<String>,
}

#[async_trait]
impl AuthSession for SigV4Session {
    async fn authenticate(&self, request: &mut HttpRequest) -> Result<()> {
        self.delegate.authenticate(request).await?;
        if let Some(prefix) = &self.catalog_prefix
            && !request.url_str().starts_with(prefix.as_str())
        {
            return Ok(());
        }
        self.signer.sign(request).await
    }
}

struct SigV4Signer {
    signing_name: String,
    signing_region: String,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    role_arn: Option<String>,
    role_session_name: Option<String>,
    /// Resolved once, so the credentials provider (and its cache) is shared by
    /// every session of the catalog.
    config: OnceCell<SdkConfig>,
}

impl Debug for SigV4Signer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SigV4Signer")
            .field("signing_name", &self.signing_name)
            .field("signing_region", &self.signing_region)
            .field("role_arn", &self.role_arn)
            .finish_non_exhaustive()
    }
}

impl SigV4Signer {
    fn from_config(cfg: &RestCatalogConfig) -> Self {
        Self {
            signing_name: cfg
                .signing_name()
                .unwrap_or_else(|| DEFAULT_SIGNING_NAME.to_string()),
            signing_region: cfg
                .signing_region()
                .unwrap_or_else(|| DEFAULT_SIGNING_REGION.to_string()),
            access_key_id: cfg.access_key_id(),
            secret_access_key: cfg.secret_access_key(),
            session_token: cfg.session_token(),
            role_arn: cfg.role_arn(),
            role_session_name: cfg.role_session_name(),
            config: OnceCell::new(),
        }
    }

    fn static_credentials(&self) -> Option<Credentials> {
        match (&self.access_key_id, &self.secret_access_key) {
            (Some(access_key_id), Some(secret_access_key)) => Some(Credentials::new(
                access_key_id,
                secret_access_key,
                self.session_token.clone(),
                None,
                "iceberg-rest-catalog",
            )),
            _ => None,
        }
    }

    async fn load_config(&self) -> SdkConfig {
        let region = Region::new(self.signing_region.clone());
        let mut config_builder =
            aws_config::defaults(BehaviorVersion::latest()).region(region.clone());

        if let Some(role_arn) = &self.role_arn {
            let role_session_name = self
                .role_session_name
                .clone()
                .unwrap_or_else(|| format!("iceberg-rest-{}", Uuid::new_v4()));
            let assume_role_builder = AssumeRoleProvider::builder(role_arn)
                .session_name(role_session_name)
                .region(region);
            // Explicit credentials, when given, are the base identity that
            // assumes the role; otherwise the default chain is.
            let assume_role_provider = match self.static_credentials() {
                Some(credentials) => {
                    assume_role_builder
                        .build_from_provider(SharedCredentialsProvider::new(credentials))
                        .await
                }
                None => assume_role_builder.build().await,
            };
            config_builder = config_builder.credentials_provider(assume_role_provider);
        } else if let Some(credentials) = self.static_credentials() {
            config_builder = config_builder.credentials_provider(credentials);
        }

        config_builder.load().await
    }

    async fn sign(&self, request: &mut HttpRequest) -> Result<()> {
        let config = self.config.get_or_init(|| self.load_config()).await;

        let credentials_provider = config.credentials_provider().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "SigV4 signing is enabled but no AWS credentials provider is configured",
            )
        })?;
        let identity = credentials_provider
            .provide_credentials()
            .await
            .map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to load AWS credentials for SigV4 signing",
                )
                .with_source(e)
            })?
            .into();

        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(&self.signing_region)
            .name(&self.signing_name)
            .time(SystemTime::now())
            .settings(SigningSettings::default())
            .build()
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "Invalid SigV4 signing parameters").with_source(e)
            })?;

        // The delegate's `Authorization` would be overwritten by the signature.
        if let Some(authorization) = request.headers_mut().remove(http::header::AUTHORIZATION) {
            request.headers_mut().insert(
                HeaderName::from_static(RELOCATED_AUTHORIZATION),
                authorization,
            );
        }

        let body = request.body().as_bytes().ok_or_else(|| {
            Error::new(
                ErrorKind::FeatureUnsupported,
                "SigV4 signing requires a buffered request body, got a streaming one",
            )
        })?;
        let headers = request
            .headers()
            .iter()
            .map(|(name, value)| {
                value
                    .to_str()
                    .map(|value| (name.as_str(), value))
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Header {name} is not valid ASCII and cannot be signed"),
                        )
                        .with_source(e)
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let signable_request = SignableRequest::new(
            request.method().as_str(),
            request.url_str(),
            headers.into_iter(),
            SignableBody::Bytes(body),
        )
        .map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                "Request cannot be signed with SigV4",
            )
            .with_source(e)
        })?;

        let (instructions, _signature) = sign(signable_request, &signing_params.into())
            .map_err(|e| Error::new(ErrorKind::Unexpected, "SigV4 signing failed").with_source(e))?
            .into_parts();
        let (signed_headers, _) = instructions.into_parts();
        for header in signed_headers {
            let mut value = HeaderValue::from_str(header.value()).map_err(|e| {
                Error::new(ErrorKind::Unexpected, "SigV4 produced an invalid header").with_source(e)
            })?;
            value.set_sensitive(header.sensitive());
            request.headers_mut().insert(header.name(), value);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::sync::LazyLock;

    use iceberg::{Catalog, CatalogBuilder};
    use reqwest::{Client, Method};
    use tokio::sync::Mutex;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::auth::{NoopAuthManager, NoopSession};
    use crate::{REST_CATALOG_PROP_URI, RestCatalogBuilder};

    /// Serializes the tests that read or mutate the AWS environment variables.
    static ENV_MUTEX: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    const ACCESS_KEY: &str = "AKIAIOSFODNN7EXAMPLE";
    const SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    const REGION: &str = "ap-northeast-2";

    fn set_env_credentials() {
        unsafe {
            env::set_var("AWS_ACCESS_KEY_ID", ACCESS_KEY);
            env::set_var("AWS_SECRET_ACCESS_KEY", SECRET_KEY);
        }
    }

    fn unset_env_credentials() {
        unsafe {
            env::remove_var("AWS_ACCESS_KEY_ID");
            env::remove_var("AWS_SECRET_ACCESS_KEY");
        }
    }

    fn sigv4_props(uri: &str, explicit_credentials: bool) -> HashMap<String, String> {
        let mut props = HashMap::from([
            (REST_CATALOG_PROP_URI.to_string(), uri.to_string()),
            ("rest.sigv4-enabled".to_string(), "true".to_string()),
            ("rest.signing-region".to_string(), REGION.to_string()),
            ("rest.signing-name".to_string(), "glue".to_string()),
        ]);
        if explicit_credentials {
            props.extend([
                ("rest.access-key-id".to_string(), ACCESS_KEY.to_string()),
                ("rest.secret-access-key".to_string(), SECRET_KEY.to_string()),
            ]);
        }
        props
    }

    fn signer(props: HashMap<String, String>) -> SigV4AuthManager {
        let cfg = RestCatalogConfig::builder()
            .uri(props[REST_CATALOG_PROP_URI].clone())
            .props(props)
            .build();
        SigV4AuthManager::new(Arc::new(NoopAuthManager), &cfg)
    }

    async fn authenticate(manager: &SigV4AuthManager, uri: &str, url: &str) -> Result<HttpRequest> {
        let session = manager.session(
            Arc::new(NoopSession),
            &HashMap::from([(REST_CATALOG_PROP_URI.to_string(), uri.to_string())]),
        );
        let mut request =
            HttpRequest::new(Client::new().request(Method::GET, url).build().unwrap());
        session.authenticate(&mut request).await?;
        Ok(request)
    }

    fn authorization(request: &HttpRequest) -> Option<&str> {
        request
            .headers()
            .get(http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
    }

    #[tokio::test]
    async fn test_signs_requests_with_explicit_credentials() {
        let uri = "http://localhost:8181";
        let manager = signer(sigv4_props(uri, true));
        let request = authenticate(&manager, uri, &format!("{uri}/v1/namespaces"))
            .await
            .unwrap();
        let header = authorization(&request).unwrap();
        assert!(header.starts_with("AWS4-HMAC-SHA256 "), "{header}");
        assert!(
            header.contains(&format!("Credential={ACCESS_KEY}/")),
            "{header}"
        );
        assert!(
            header.contains(&format!("/{REGION}/glue/aws4_request")),
            "{header}"
        );
    }

    #[tokio::test]
    async fn test_signs_requests_with_env_credentials() {
        let _guard = ENV_MUTEX.lock().await;
        set_env_credentials();
        let uri = "http://localhost:8181";
        let manager = signer(sigv4_props(uri, false));
        let request = authenticate(&manager, uri, &format!("{uri}/v1/config")).await;
        unset_env_credentials();
        let header = authorization(&request.unwrap()).unwrap().to_string();
        assert!(header.starts_with("AWS4-HMAC-SHA256 "), "{header}");
    }

    #[tokio::test]
    async fn test_explicit_credentials_override_env() {
        let _guard = ENV_MUTEX.lock().await;
        set_env_credentials();
        let uri = "http://localhost:8181";
        let mut props = sigv4_props(uri, false);
        props.extend([
            ("rest.access-key-id".to_string(), "EXPLICIT_KEY".to_string()),
            (
                "rest.secret-access-key".to_string(),
                "EXPLICIT_SECRET".to_string(),
            ),
        ]);
        let manager = signer(props);
        let request = authenticate(&manager, uri, &format!("{uri}/v1/config")).await;
        unset_env_credentials();
        let header = authorization(&request.unwrap()).unwrap().to_string();
        assert!(header.contains("Credential=EXPLICIT_KEY/"), "{header}");
    }

    #[tokio::test]
    async fn test_fails_without_credentials() {
        let _guard = ENV_MUTEX.lock().await;
        // Point every source of the default credentials chain at nothing, so
        // the result does not depend on the machine running the test.
        let isolated = [
            (
                "AWS_SHARED_CREDENTIALS_FILE",
                Some("/nonexistent/credentials"),
            ),
            ("AWS_CONFIG_FILE", Some("/nonexistent/config")),
            ("AWS_EC2_METADATA_DISABLED", Some("true")),
            ("AWS_ACCESS_KEY_ID", None),
            ("AWS_SECRET_ACCESS_KEY", None),
            ("AWS_SESSION_TOKEN", None),
            ("AWS_PROFILE", None),
            ("AWS_WEB_IDENTITY_TOKEN_FILE", None),
            ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", None),
            ("AWS_CONTAINER_CREDENTIALS_FULL_URI", None),
        ];
        let saved: Vec<_> = isolated
            .iter()
            .map(|(name, _)| (*name, env::var_os(name)))
            .collect();
        for (name, value) in isolated {
            unsafe {
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
        }

        let uri = "http://localhost:8181";
        let manager = signer(sigv4_props(uri, false));
        let result = authenticate(&manager, uri, &format!("{uri}/v1/config")).await;

        for (name, value) in saved {
            unsafe {
                match value {
                    Some(value) => env::set_var(name, value),
                    None => env::remove_var(name),
                }
            }
        }
        let Err(err) = result else {
            panic!("signing without credentials must fail");
        };
        assert_eq!(err.kind(), ErrorKind::Unexpected, "{err}");
        assert!(err.to_string().contains("AWS credentials"), "{err}");
    }

    #[tokio::test]
    async fn test_skips_requests_outside_the_catalog() {
        let uri = "http://localhost:8181";
        let manager = signer(sigv4_props(uri, true));
        let request = authenticate(&manager, uri, "http://other-host/v1/oauth/tokens")
            .await
            .unwrap();
        assert!(authorization(&request).is_none());
    }

    #[tokio::test]
    async fn test_relocates_delegate_authorization() {
        #[derive(Debug)]
        struct BearerSession;
        #[async_trait]
        impl AuthSession for BearerSession {
            async fn authenticate(&self, request: &mut HttpRequest) -> Result<()> {
                request.headers_mut().insert(
                    http::header::AUTHORIZATION,
                    HeaderValue::from_static("Bearer token"),
                );
                Ok(())
            }
        }

        let uri = "http://localhost:8181";
        let manager = signer(sigv4_props(uri, true));
        let session = manager.session(
            Arc::new(BearerSession),
            &HashMap::from([(REST_CATALOG_PROP_URI.to_string(), uri.to_string())]),
        );
        let mut request = HttpRequest::new(
            Client::new()
                .request(Method::GET, format!("{uri}/v1/namespaces"))
                .build()
                .unwrap(),
        );
        session.authenticate(&mut request).await.unwrap();
        assert!(
            authorization(&request)
                .unwrap()
                .starts_with("AWS4-HMAC-SHA256 ")
        );
        assert_eq!(
            request.headers().get(RELOCATED_AUTHORIZATION).unwrap(),
            "Bearer token"
        );
    }

    /// A stub REST catalog that answers the config fetch and a namespace listing.
    async fn stub_catalog() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/config"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "defaults": {},
                "overrides": {},
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/namespaces"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "namespaces": [] })),
            )
            .mount(&server)
            .await;
        server
    }

    /// One entry per request the stub saw, `None` where it carried no
    /// `Authorization`, so a catalog signing only some requests is caught.
    async fn authorization_headers(
        server: &MockServer,
        props: HashMap<String, String>,
    ) -> Vec<Option<String>> {
        let catalog = RestCatalogBuilder::default()
            .load("rest", props)
            .await
            .unwrap();
        catalog.list_namespaces(None).await.unwrap();
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .map(|request| {
                request
                    .headers
                    .get("authorization")
                    .and_then(|value| value.to_str().ok())
                    .map(ToString::to_string)
            })
            .collect()
    }

    #[tokio::test]
    async fn test_catalog_signs_every_request() {
        let server = stub_catalog().await;
        let requests = authorization_headers(&server, sigv4_props(&server.uri(), true)).await;
        assert_eq!(requests.len(), 2, "{requests:?}");
        for header in &requests {
            let header = header.as_deref().expect("every request is signed");
            assert!(header.starts_with("AWS4-HMAC-SHA256 "), "{header}");
            assert!(
                header.contains(&format!("Credential={ACCESS_KEY}/")),
                "{header}"
            );
            assert!(
                header.contains(&format!("/{REGION}/glue/aws4_request")),
                "{header}"
            );
        }
    }

    #[tokio::test]
    async fn test_catalog_auth_type_sigv4_signs_every_request() {
        let server = stub_catalog().await;
        let mut props = sigv4_props(&server.uri(), true);
        props.remove("rest.sigv4-enabled");
        props.insert("rest.auth.type".to_string(), AUTH_TYPE_SIGV4.to_string());
        let requests = authorization_headers(&server, props).await;
        assert_eq!(requests.len(), 2, "{requests:?}");
        assert!(
            requests.iter().all(|h| h
                .as_deref()
                .is_some_and(|h| h.starts_with("AWS4-HMAC-SHA256 "))),
            "{requests:?}"
        );
    }

    #[tokio::test]
    async fn test_catalog_without_sigv4_sends_no_signature() {
        let server = stub_catalog().await;
        let props = HashMap::from([(REST_CATALOG_PROP_URI.to_string(), server.uri())]);
        let requests = authorization_headers(&server, props).await;
        assert!(!requests.is_empty());
        assert!(requests.iter().all(Option::is_none), "{requests:?}");
    }

    #[tokio::test]
    #[ignore] // Requires AWS credentials and a valid role ARN.
    async fn test_signs_with_assumed_role() {
        let uri = "http://localhost:8181";
        let role_arn = env::var("ICEBERG_TEST_ROLE_ARN").expect("ICEBERG_TEST_ROLE_ARN is not set");
        let mut props = sigv4_props(uri, false);
        props.extend([
            ("rest.client.assume-role.arn".to_string(), role_arn),
            (
                "rest.client.assume-role.session-name".to_string(),
                "test-session-name".to_string(),
            ),
        ]);
        let manager = signer(props);
        let request = authenticate(&manager, uri, &format!("{uri}/v1/config"))
            .await
            .unwrap();
        assert!(
            authorization(&request)
                .unwrap()
                .starts_with("AWS4-HMAC-SHA256 ")
        );
    }

    /// End-to-end against the AWS Glue Iceberg REST endpoint: AWS accepts a
    /// request only if its signature is valid, so this covers what the stub
    /// tests cannot — GETs, POSTs with a signed JSON body, and DELETEs.
    ///
    /// Uses the default AWS credentials chain. Creates and drops a uniquely
    /// named namespace (and a table when `ICEBERG_GLUE_TEST_LOCATION` is set;
    /// dropping it does not purge the metadata file Glue writes there).
    ///
    /// ```text
    /// ICEBERG_GLUE_TEST_REGION=us-west-2 ICEBERG_GLUE_TEST_ACCOUNT_ID=123456789012 \
    /// ICEBERG_GLUE_TEST_LOCATION=s3://bucket/prefix \
    /// cargo test -p iceberg-catalog-rest --lib test_glue_end_to_end -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore] // Requires AWS credentials and access to AWS Glue.
    async fn test_glue_end_to_end() {
        use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
        use iceberg::{NamespaceIdent, TableCreation};

        let region = env::var("ICEBERG_GLUE_TEST_REGION").expect("ICEBERG_GLUE_TEST_REGION");
        let account_id =
            env::var("ICEBERG_GLUE_TEST_ACCOUNT_ID").expect("ICEBERG_GLUE_TEST_ACCOUNT_ID");
        let location = env::var("ICEBERG_GLUE_TEST_LOCATION").ok();

        // Table objects need a FileIO; nothing is read or written through it.
        let catalog = RestCatalogBuilder::default()
            .with_storage_factory(Arc::new(iceberg::io::MemoryStorageFactory))
            .load(
                "glue",
                HashMap::from([
                    (
                        REST_CATALOG_PROP_URI.to_string(),
                        format!("https://glue.{region}.amazonaws.com/iceberg"),
                    ),
                    ("warehouse".to_string(), account_id),
                    ("rest.sigv4-enabled".to_string(), "true".to_string()),
                    ("rest.signing-region".to_string(), region),
                    ("rest.signing-name".to_string(), "glue".to_string()),
                ]),
            )
            .await
            .unwrap();

        let namespace = NamespaceIdent::new(format!(
            "iceberg_rust_sigv4_e2e_{}",
            &Uuid::new_v4().simple().to_string()[..8]
        ));
        let table = iceberg::TableIdent::new(namespace.clone(), "t".to_string());

        // GET /v1/config, GET /v1/namespaces
        catalog.list_namespaces(None).await.unwrap();
        // POST /v1/namespaces with a JSON body
        catalog
            .create_namespace(
                &namespace,
                HashMap::from([("comment".to_string(), "iceberg-rust sigv4 e2e".to_string())]),
            )
            .await
            .unwrap();
        println!("created namespace {namespace:?}");

        let result = async {
            // GET /v1/namespaces/{ns}
            catalog.get_namespace(&namespace).await?;
            if let Some(location) = &location {
                let schema = Schema::builder()
                    .with_fields(vec![
                        NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    ])
                    .build()?;
                // POST /v1/namespaces/{ns}/tables with a larger JSON body
                catalog
                    .create_table(
                        &namespace,
                        TableCreation::builder()
                            .name(table.name().to_string())
                            .location(format!("{location}/{}/t", namespace.to_url_string()))
                            .schema(schema)
                            .build(),
                    )
                    .await?;
                println!("created table {table}");
                // GET /v1/namespaces/{ns}/tables/{t}
                catalog.load_table(&table).await?;
                // DELETE /v1/namespaces/{ns}/tables/{t}
                catalog.drop_table(&table).await?;
            }
            Ok::<_, Error>(())
        }
        .await;

        // DELETE /v1/namespaces/{ns}, attempted even if a step above failed.
        if result.is_err() && location.is_some() {
            let _ = catalog.drop_table(&table).await;
        }
        let dropped = catalog.drop_namespace(&namespace).await;
        result.unwrap();
        dropped.unwrap();
        println!("dropped namespace {namespace:?}");
    }
}
