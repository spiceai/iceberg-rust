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

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{DataFile, FormatVersion, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind};

/// `RowDeltaAction` is a transaction action for adding both data files and delete
/// files to a table in a single atomic operation.
///
/// This action produces a snapshot with `Operation::Overwrite` when both data and
/// delete files are present, or `Operation::Delete` when only delete files are added.
///
/// # Example
///
/// ```ignore
/// use iceberg::transaction::{ApplyTransactionAction, Transaction};
///
/// let tx = Transaction::new(&table);
/// let action = tx.row_delta()
///     .add_delete_files(equality_delete_files)
///     .add_data_files(new_data_files);
/// let tx = action.apply(tx)?;
/// let table = tx.commit(&catalog).await?;
/// ```
pub struct RowDeltaAction {
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    added_delete_files: Vec<DataFile>,
}

impl RowDeltaAction {
    pub(crate) fn new() -> Self {
        Self {
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            added_data_files: vec![],
            added_delete_files: vec![],
        }
    }

    /// Add data files to the snapshot.
    pub fn add_data_files(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Add delete files (equality or position) to the snapshot.
    pub fn add_delete_files(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_delete_files.extend(delete_files);
        self
    }

    /// Set commit UUID for the snapshot.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Set key metadata for manifest files.
    pub fn set_key_metadata(mut self, key_metadata: Vec<u8>) -> Self {
        self.key_metadata = Some(key_metadata);
        self
    }

    /// Set snapshot summary properties.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }
}

#[async_trait]
impl TransactionAction for RowDeltaAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if self.added_data_files.is_empty() && self.added_delete_files.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Row delta requires at least one data file or delete file",
            ));
        }

        // Row delta requires format version >= 2 for delete files
        if !self.added_delete_files.is_empty()
            && table.metadata().format_version() == FormatVersion::V1
        {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Delete files are not supported in format version 1",
            ));
        }

        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
        )
        .with_delete_files(self.added_delete_files.clone());

        // Validate added data files
        if !self.added_data_files.is_empty() {
            snapshot_producer.validate_added_data_files()?;
        }

        // Validate added delete files
        if !self.added_delete_files.is_empty() {
            snapshot_producer.validate_added_delete_files()?;
        }

        let has_data_files = !self.added_data_files.is_empty();
        let operation = RowDeltaOperation { has_data_files };

        snapshot_producer
            .commit(operation, DefaultManifestProcess)
            .await
    }
}

struct RowDeltaOperation {
    has_data_files: bool,
}

impl SnapshotProduceOperation for RowDeltaOperation {
    fn operation(&self) -> Operation {
        if self.has_data_files {
            // When both data and delete files are added, the operation is Overwrite
            Operation::Overwrite
        } else {
            // When only delete files are added, the operation is Delete
            Operation::Delete
        }
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        // Carry forward all existing manifests (both data and delete) from the current snapshot
        let Some(snapshot) = snapshot_produce.table.metadata().current_snapshot() else {
            return Ok(vec![]);
        };

        let manifest_list = snapshot
            .load_manifest_list(
                snapshot_produce.table.file_io(),
                &snapshot_produce.table.metadata_ref(),
            )
            .await?;

        Ok(manifest_list
            .entries()
            .iter()
            .filter(|entry| entry.has_added_files() || entry.has_existing_files())
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, MAIN_BRANCH,
        ManifestContentType, Struct,
    };
    use crate::transaction::tests::{make_v1_table, make_v2_minimal_table};
    use crate::transaction::{Transaction, TransactionAction};
    use crate::{TableRequirement, TableUpdate};

    fn make_data_file(table: &crate::table::Table, path: &str) -> crate::spec::DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .build()
            .unwrap()
    }

    fn make_equality_delete_file(table: &crate::table::Table, path: &str) -> crate::spec::DataFile {
        DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(50)
            .record_count(2)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .equality_ids(Some(vec![1]))
            .build()
            .unwrap()
    }

    fn make_position_delete_file(table: &crate::table::Table, path: &str) -> crate::spec::DataFile {
        DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(50)
            .record_count(3)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_row_delta_empty_files() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);
        let action = tx.row_delta();
        assert!(Arc::new(action).commit(&table).await.is_err());
    }

    #[tokio::test]
    async fn test_row_delta_delete_only() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let delete_file = make_equality_delete_file(&table, "test/eq-del-1.parquet");
        let action = tx.row_delta().add_delete_files(vec![delete_file.clone()]);

        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();
        let requirements = action_commit.take_requirements();

        // Check updates structure
        assert!(matches!((&updates[0], &updates[1]),
                (TableUpdate::AddSnapshot { snapshot }, TableUpdate::SetSnapshotRef { reference, ref_name })
                if snapshot.snapshot_id() == reference.snapshot_id && ref_name == MAIN_BRANCH));

        // Check operation is Delete when only delete files present
        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };
        assert_eq!(
            new_snapshot.summary().operation,
            crate::spec::Operation::Delete
        );

        // Check requirements
        assert_eq!(
            vec![
                TableRequirement::UuidMatch {
                    uuid: table.metadata().uuid()
                },
                TableRequirement::RefSnapshotIdMatch {
                    r#ref: MAIN_BRANCH.to_string(),
                    snapshot_id: table.metadata().current_snapshot_id
                }
            ],
            requirements
        );

        // Check manifest list: should have 1 manifest (the delete manifest)
        let manifest_list = new_snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert_eq!(1, manifest_list.entries().len());

        // The manifest should be a delete manifest
        assert_eq!(
            manifest_list.entries()[0].content,
            ManifestContentType::Deletes
        );

        // Check manifest entries
        let manifest = manifest_list.entries()[0]
            .load_manifest(table.file_io())
            .await
            .unwrap();
        assert_eq!(1, manifest.entries().len());
        assert_eq!(delete_file, *manifest.entries()[0].data_file());
    }

    #[tokio::test]
    async fn test_row_delta_data_and_delete() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let data_file = make_data_file(&table, "test/data-1.parquet");
        let delete_file = make_equality_delete_file(&table, "test/eq-del-1.parquet");

        let action = tx
            .row_delta()
            .add_data_files(vec![data_file.clone()])
            .add_delete_files(vec![delete_file.clone()]);

        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        // Check operation is Overwrite when both data and delete files present
        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };
        assert_eq!(
            new_snapshot.summary().operation,
            crate::spec::Operation::Overwrite
        );

        // Check manifest list: should have 2 manifests (data + delete)
        let manifest_list = new_snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert_eq!(2, manifest_list.entries().len());

        // Find the data manifest and delete manifest
        let data_manifests: Vec<_> = manifest_list
            .entries()
            .iter()
            .filter(|e| e.content == ManifestContentType::Data)
            .collect();
        let delete_manifests: Vec<_> = manifest_list
            .entries()
            .iter()
            .filter(|e| e.content == ManifestContentType::Deletes)
            .collect();
        assert_eq!(1, data_manifests.len());
        assert_eq!(1, delete_manifests.len());

        // Verify data manifest contents
        let data_manifest = data_manifests[0]
            .load_manifest(table.file_io())
            .await
            .unwrap();
        assert_eq!(1, data_manifest.entries().len());
        assert_eq!(data_file, *data_manifest.entries()[0].data_file());

        // Verify delete manifest contents
        let delete_manifest = delete_manifests[0]
            .load_manifest(table.file_io())
            .await
            .unwrap();
        assert_eq!(1, delete_manifest.entries().len());
        assert_eq!(delete_file, *delete_manifest.entries()[0].data_file());
    }

    #[tokio::test]
    async fn test_row_delta_position_delete() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let delete_file = make_position_delete_file(&table, "test/pos-del-1.parquet");
        let action = tx.row_delta().add_delete_files(vec![delete_file.clone()]);

        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };

        let manifest_list = new_snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert_eq!(1, manifest_list.entries().len());
        assert_eq!(
            manifest_list.entries()[0].content,
            ManifestContentType::Deletes
        );

        let manifest = manifest_list.entries()[0]
            .load_manifest(table.file_io())
            .await
            .unwrap();
        assert_eq!(1, manifest.entries().len());
        assert_eq!(
            manifest.entries()[0].data_file().content_type(),
            DataContentType::PositionDeletes
        );
    }

    #[tokio::test]
    async fn test_row_delta_v1_rejects_delete_files() {
        let table = make_v1_table();

        // V1 should reject delete files
        let tx = Transaction::new(&table);
        let delete_file = DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path("test/eq-del-1.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(50)
            .record_count(1)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .equality_ids(Some(vec![1]))
            .build()
            .unwrap();
        let action = tx.row_delta().add_delete_files(vec![delete_file]);
        let result = Arc::new(action).commit(&table).await;
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("Expected error for V1 delete files"),
        };
        assert!(err_msg.contains("not supported in format version 1"));
    }

    #[tokio::test]
    async fn test_row_delta_rejects_data_content_in_delete_files() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        // Try to add a data file as a delete file — should fail validation
        let bad_delete_file = make_data_file(&table, "test/not-a-delete.parquet");
        let action = tx.row_delta().add_delete_files(vec![bad_delete_file]);
        let result = Arc::new(action).commit(&table).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_row_delta_snapshot_properties() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let mut props = HashMap::new();
        props.insert("custom-key".to_string(), "custom-value".to_string());

        let delete_file = make_equality_delete_file(&table, "test/eq-del-1.parquet");
        let action = tx
            .row_delta()
            .add_delete_files(vec![delete_file])
            .set_snapshot_properties(props);

        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };
        assert_eq!(
            new_snapshot
                .summary()
                .additional_properties
                .get("custom-key")
                .unwrap(),
            "custom-value"
        );
    }

    #[tokio::test]
    async fn test_row_delta_multiple_delete_files() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let eq_del = make_equality_delete_file(&table, "test/eq-del-1.parquet");
        let pos_del = make_position_delete_file(&table, "test/pos-del-1.parquet");

        let action = tx
            .row_delta()
            .add_delete_files(vec![eq_del.clone(), pos_del.clone()]);

        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };

        let manifest_list = new_snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert_eq!(1, manifest_list.entries().len());
        assert_eq!(
            manifest_list.entries()[0].content,
            ManifestContentType::Deletes
        );

        let manifest = manifest_list.entries()[0]
            .load_manifest(table.file_io())
            .await
            .unwrap();
        assert_eq!(2, manifest.entries().len());
    }

    #[tokio::test]
    async fn test_row_delta_summary_metrics() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let data_file = make_data_file(&table, "test/data-1.parquet");
        let eq_del = make_equality_delete_file(&table, "test/eq-del-1.parquet");

        let action = tx
            .row_delta()
            .add_data_files(vec![data_file])
            .add_delete_files(vec![eq_del]);

        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = action_commit.take_updates();

        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            unreachable!()
        };

        let props = &new_snapshot.summary().additional_properties;

        // Should have added-data-files = 1
        assert_eq!(props.get("added-data-files").unwrap(), "1");
        // Should have added-equality-delete-files = 1
        assert_eq!(props.get("added-equality-delete-files").unwrap(), "1");
        // Should have added-delete-files = 1
        assert_eq!(props.get("added-delete-files").unwrap(), "1");
    }
}
