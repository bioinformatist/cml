#[allow(dead_code)]
#[repr(u8)]
pub enum ReplicaNum {
    NoReplica = 1,
    WithReplica = 3,
}

#[allow(dead_code)]
pub enum CacheModel {
    None,
    LastRow,
    LastValue,
    Both,
}

impl CacheModel {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            CacheModel::None => "none",
            CacheModel::LastRow => "last_row",
            CacheModel::LastValue => "last_value",
            CacheModel::Both => "both",
        }
    }
}

#[allow(dead_code)]
#[repr(u8)]
pub enum SingleSTable {
    False,
    True,
}
