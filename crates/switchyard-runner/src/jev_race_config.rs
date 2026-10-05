// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit route settings for the JEV race and its optional request recording.

use std::{path::PathBuf, sync::Arc, time::Duration};

use serde::Deserialize;
use switchyard_llm_client::jev_race::{JevRaceClient, JevRaceConfig};
use switchyard_protocol::RoutedLlmClient;

use crate::RunnerError;

/// The race is enabled only on routes that include this table.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct JevRaceRouteConfig {
    enabled: bool,
    observe_only: bool,
    evaluation_wait_both: bool,
    assumed_jev_latency_ms: u64,
    service_timeout_ms: u64,
    disable_hold: bool,
    endpoint: String,
    api_key_env: String,
    model: String,
    threshold: f64,
    max_hold_ms: u64,
    observed_retail_ids: bool,
    observed_domain_ids: bool,
    audit_directory: Option<PathBuf>,
}

impl Default for JevRaceRouteConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            observe_only: false,
            evaluation_wait_both: false,
            assumed_jev_latency_ms: 200,
            service_timeout_ms: 30_000,
            disable_hold: false,
            endpoint: "https://api.typesafe.ai/v1/systemone".to_string(),
            api_key_env: "JEV_KEY".to_string(),
            model: "jev-1.13.0".to_string(),
            threshold: 0.9,
            max_hold_ms: 400,
            observed_retail_ids: false,
            observed_domain_ids: false,
            audit_directory: None,
        }
    }
}

impl JevRaceRouteConfig {
    /// Disabled routes record baseline traffic without requiring a JEV credential.
    pub(crate) fn api_key(&self) -> Result<String, RunnerError> {
        if !self.enabled {
            return Ok(String::new());
        }
        if self.api_key_env.trim().is_empty() {
            return Err(RunnerError::configuration(
                "jev_race api_key_env must not be empty",
            ));
        }
        let key = std::env::var(&self.api_key_env).map_err(|_| {
            RunnerError::configuration(format!(
                "jev_race could not read api_key_env {}",
                self.api_key_env
            ))
        })?;
        if key.trim().is_empty() {
            return Err(RunnerError::configuration(format!(
                "jev_race api_key_env {} is empty",
                self.api_key_env
            )));
        }
        Ok(key)
    }

    /// Wraps only the configured completion client; the original model request is retained.
    pub(crate) fn wrap(
        &self,
        upstream: Arc<dyn RoutedLlmClient>,
        api_key: &str,
        redaction_keys: &[String],
    ) -> Result<Arc<dyn RoutedLlmClient>, RunnerError> {
        let config = JevRaceConfig {
            endpoint: self.endpoint.clone(),
            model: self.model.clone(),
            threshold: self.threshold,
            deadline: (!self.disable_hold).then(|| Duration::from_millis(self.max_hold_ms)),
            observed_retail_ids: self.observed_retail_ids,
            observed_domain_ids: self.observed_domain_ids,
            audit_directory: self.audit_directory.clone(),
            enabled: self.enabled,
            observe_only: self.observe_only,
            evaluation_wait_both: self.evaluation_wait_both,
            assumed_jev_latency: Duration::from_millis(self.assumed_jev_latency_ms),
            service_timeout: Duration::from_millis(self.service_timeout_ms),
        };
        let client = JevRaceClient::new(upstream, config, api_key.to_string())?
            .with_redaction_keys(redaction_keys.to_vec());
        Ok(Arc::new(client))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_bound_the_stream_hold_and_do_not_enable_retail_adapters() {
        let config: JevRaceRouteConfig = toml::from_str("").expect("empty explicit table");
        assert!(config.enabled);
        assert!(!config.observe_only);
        assert_eq!(config.max_hold_ms, 400);
        assert_eq!(config.threshold, 0.9);
        assert!(!config.observed_retail_ids);
        assert!(config.audit_directory.is_none());
    }

    #[test]
    fn parses_wait_both_with_separate_operational_timeout() {
        let config: JevRaceRouteConfig = toml::from_str(
            "evaluation_wait_both = true\ndisable_hold = true\nservice_timeout_ms = 60000",
        )
        .unwrap();
        assert!(config.evaluation_wait_both);
        assert!(config.disable_hold);
        assert_eq!(config.assumed_jev_latency_ms, 200);
        assert_eq!(config.service_timeout_ms, 60_000);
    }

    #[test]
    fn baseline_does_not_read_a_missing_key() {
        let config: JevRaceRouteConfig =
            toml::from_str("enabled = false\napi_key_env = 'SWITCHYARD_JEV_TEST_UNSET_KEY'")
                .expect("baseline settings");
        assert_eq!(config.api_key().expect("disabled credential"), "");
    }

    #[test]
    fn rejects_misspelled_settings() {
        assert!(toml::from_str::<JevRaceRouteConfig>("max_hlod_ms = 400").is_err());
    }
}
