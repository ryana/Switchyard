// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build finite, complete tool calls from the current model request.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::{Map, Value};

/// Reserve one of the service's 255 choice labels for NONE.
pub const MAX_OPTIONS: usize = 254;

#[derive(Clone, Debug, Serialize)]
pub struct CompleteCall {
    pub name: String,
    pub arguments: Value,
    pub description: String,
    pub origin: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Eligibility {
    pub tool: String,
    pub status: String,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ChoiceSet {
    pub options: BTreeMap<String, CompleteCall>,
    pub audit: Vec<Eligibility>,
}

impl ChoiceSet {
    fn note(&mut self, tool: &str, status: &str, reason: impl Into<String>) {
        self.audit.push(Eligibility {
            tool: tool.into(),
            status: status.into(),
            reason: reason.into(),
        });
    }
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn keys(schema: &Map<String, Value>, allowed: &[&str]) -> Result<(), String> {
    if let Some(key) = schema.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("Unsupported schema keyword: {key}"));
    }
    Ok(())
}

fn integer(value: &Value) -> bool {
    value.is_i64() || value.is_u64()
}

fn scalar(value: &Value) -> bool {
    value.is_string() || value.is_boolean() || integer(value)
}

fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "string" => value.is_string(),
        "integer" => integer(value),
        "boolean" => value.is_boolean(),
        _ => false,
    }
}

/// Interpret only constraints we can fully check; never ignore a constraint.
fn finite_values(value: &Value) -> Result<Vec<Value>, String> {
    let schema = object(value, "Argument schema")?;
    keys(
        schema,
        &[
            "type",
            "enum",
            "description",
            "title",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
        ],
    )?;
    let kind = match schema.get("type") {
        None => None,
        Some(Value::String(s)) if ["string", "integer", "boolean"].contains(&s.as_str()) => {
            Some(s.as_str())
        }
        _ => return Err("Only string, integer, and boolean argument types are supported".into()),
    };
    let numeric = [
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    ];
    for key in numeric {
        if let Some(value) = schema.get(key) {
            if kind != Some("integer") || value.as_i64().is_none() {
                return Err(format!(
                    "{key} requires an integer type and signed integer constraint"
                ));
            }
            if key == "multipleOf" && value.as_i64().unwrap() <= 0 {
                return Err("multipleOf must be positive".into());
            }
        }
    }
    for key in ["minLength", "maxLength"] {
        if let Some(value) = schema.get(key)
            && (kind != Some("string") || value.as_u64().is_none())
        {
            return Err(format!(
                "{key} requires a string type and nonnegative integer constraint"
            ));
        }
    }
    let values = if let Some(values) = schema.get("enum") {
        let values = values.as_array().ok_or("enum must be an array")?;
        if values.is_empty() || values.iter().any(|v| !scalar(v)) {
            return Err("enum must contain string, integer, or boolean choices".into());
        }
        values.clone()
    } else if kind == Some("boolean") {
        vec![Value::Bool(false), Value::Bool(true)]
    } else if kind == Some("integer") {
        let lo = schema
            .get("minimum")
            .and_then(Value::as_i64)
            .ok_or("Integer needs explicit minimum")?;
        let hi = schema
            .get("maximum")
            .and_then(Value::as_i64)
            .ok_or("Integer needs explicit maximum")?;
        if hi < lo || i128::from(hi) - i128::from(lo) >= MAX_OPTIONS as i128 {
            return Err("Integer range exceeds the finite choice limit or is empty".into());
        }
        (lo..=hi).map(Value::from).collect()
    } else {
        return Err("Open text is not a finite argument".into());
    };
    let mut result = Vec::new();
    for value in values {
        if kind.is_some_and(|kind| !matches_type(&value, kind)) {
            continue;
        }
        if integer(&value) {
            let number = value
                .as_i64()
                .map(i128::from)
                .unwrap_or_else(|| i128::from(value.as_u64().unwrap()));
            let bound = |key: &str| schema.get(key).and_then(Value::as_i64).map(i128::from);
            if bound("minimum").is_some_and(|v| number < v)
                || bound("maximum").is_some_and(|v| number > v)
                || bound("exclusiveMinimum").is_some_and(|v| number <= v)
                || bound("exclusiveMaximum").is_some_and(|v| number >= v)
                || bound("multipleOf").is_some_and(|v| number % v != 0)
            {
                continue;
            }
        }
        if let Some(text) = value.as_str() {
            let length = text.chars().count() as u64;
            if schema
                .get("minLength")
                .and_then(Value::as_u64)
                .is_some_and(|v| length < v)
                || schema
                    .get("maxLength")
                    .and_then(Value::as_u64)
                    .is_some_and(|v| length > v)
            {
                continue;
            }
        }
        if !result.contains(&value) {
            result.push(value);
        }
    }
    if result.is_empty() {
        return Err("No enum or range value satisfies all supported constraints".into());
    }
    Ok(result)
}

fn lookup_argument(name: &str) -> Option<&'static str> {
    match name {
        "get_user_details" => Some("user_id"),
        "get_order_details" => Some("order_id"),
        "get_product_details" => Some("product_id"),
        _ => None,
    }
}

fn observe(value: &Value, ids: &mut BTreeMap<String, BTreeSet<String>>) {
    match value {
        Value::Object(map) => {
            for (key, value) in map {
                if let Some(set) = ids.get_mut(key)
                    && let Some(id) = value.as_str().filter(|id| !id.is_empty())
                {
                    set.insert(id.into());
                }
                if key == "orders"
                    && let Some(orders) = value.as_array()
                {
                    for id in orders
                        .iter()
                        .filter_map(Value::as_str)
                        .filter(|id| !id.is_empty())
                    {
                        ids.get_mut("order_id").unwrap().insert(id.into());
                    }
                }
                observe(value, ids);
            }
        }
        Value::Array(items) => items.iter().for_each(|item| observe(item, ids)),
        _ => {}
    }
}

fn seen_ids(request: &Value) -> BTreeMap<String, BTreeSet<String>> {
    let mut ids: BTreeMap<String, BTreeSet<String>> = ["user_id", "order_id", "product_id"]
        .into_iter()
        .map(|key| (key.into(), BTreeSet::new()))
        .collect();
    let mut calls = BTreeMap::<String, String>::new();
    for message in request["messages"].as_array().into_iter().flatten() {
        if message["role"] == "assistant" {
            for call in message["tool_calls"].as_array().into_iter().flatten() {
                if let (Some(id), Some(name)) =
                    (call["id"].as_str(), call["function"]["name"].as_str())
                {
                    calls.insert(id.into(), name.into());
                }
            }
        }
        if message["role"] != "tool" || message["is_error"] == true {
            continue;
        }
        let Some(content) = message["content"].as_str() else {
            continue;
        };
        if content.trim().is_empty() || content.trim_start().starts_with("Error:") {
            continue;
        }
        let linked = message["tool_call_id"]
            .as_str()
            .and_then(|id| calls.get(id))
            .map(String::as_str);
        let explicit = message["name"].as_str();
        if explicit.zip(linked).is_some_and(|(a, b)| a != b) {
            continue;
        }
        let name = explicit.or(linked);
        if matches!(
            name,
            Some("find_user_id_by_email" | "find_user_id_by_name_zip")
        ) {
            // Retail's lookup result is raw text; translated sources can JSON-quote it.
            match serde_json::from_str::<Value>(content) {
                Ok(Value::String(id)) if !id.is_empty() => {
                    ids.get_mut("user_id").unwrap().insert(id);
                }
                Err(_) => {
                    ids.get_mut("user_id").unwrap().insert(content.into());
                }
                _ => {}
            }
        } else if let Ok(value) = serde_json::from_str::<Value>(content) {
            observe(&value, &mut ids);
            if name == Some("list_all_product_types")
                && let Some(products) = value.as_object()
            {
                for id in products
                    .values()
                    .filter_map(Value::as_str)
                    .filter(|id| !id.is_empty())
                {
                    ids.get_mut("product_id").unwrap().insert(id.into());
                }
            }
        }
    }
    ids
}

fn tool_calls(
    tool: &Value,
    known: &BTreeMap<String, BTreeSet<String>>,
    adapt: bool,
) -> Result<(Vec<Value>, &'static str), String> {
    let schema = object(&tool["parameters"], "Parameter schema")?;
    keys(
        schema,
        &[
            "type",
            "properties",
            "required",
            "additionalProperties",
            "description",
            "title",
        ],
    )?;
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err("Parameter schema must have object type".into());
    }
    if schema
        .get("additionalProperties")
        .is_some_and(|value| value != &Value::Bool(false))
    {
        return Err(
            "Explicitly open or schema-valued additional properties are unsupported".into(),
        );
    }
    let empty = Map::new();
    let properties = match schema.get("properties") {
        Some(value) => object(value, "properties")?,
        None => &empty,
    };
    let mut required = BTreeSet::new();
    if let Some(value) = schema.get("required") {
        for name in value.as_array().ok_or("required must be an array")? {
            let name = name.as_str().ok_or("required names must be strings")?;
            if !properties.contains_key(name) || !required.insert(name) {
                return Err("Required names must be unique and have property schemas".into());
            }
        }
    }
    let mut combinations = vec![Map::new()];
    let mut origin = "native";
    for (name, spec) in properties {
        let values = match finite_values(spec) {
            Ok(values) => values,
            Err(error) => {
                if !adapt
                    || lookup_argument(tool["name"].as_str().unwrap_or_default())
                        != Some(name.as_str())
                    || properties.len() != 1
                {
                    return Err(format!("{name}: {error}"));
                }
                let spec = object(spec, "Identifier schema")?;
                keys(spec, &["type", "description", "title"])?;
                if spec.get("type").and_then(Value::as_str) != Some("string") {
                    return Err(
                        "Observed-ID adaptation requires an unconstrained string identifier".into(),
                    );
                }
                let values: Vec<Value> = known[name].iter().cloned().map(Value::String).collect();
                if values.is_empty() {
                    return Err("No identifier observed in earlier tool results".into());
                }
                origin = "observed_id_choices";
                values
            }
        };
        let optional = !required.contains(name.as_str());
        if combinations
            .len()
            .checked_mul(values.len() + usize::from(optional))
            .is_none_or(|n| n > MAX_OPTIONS)
        {
            return Err(
                "Complete option set exceeds the choice limit; no partial truncation".into(),
            );
        }
        let mut next = Vec::new();
        for arguments in combinations {
            if optional {
                next.push(arguments.clone());
            }
            for value in &values {
                let mut arguments = arguments.clone();
                arguments.insert(name.clone(), value.clone());
                next.push(arguments);
            }
        }
        combinations = next;
    }
    Ok((
        combinations.into_iter().map(Value::Object).collect(),
        origin,
    ))
}

/// Build options from an OpenAI Chat request. NONE is added by the caller.
///
/// The retail adapter is deliberately opt-in. It only reuses identifiers in prior
/// tool results and cannot read the store, task description, or future answers.
pub fn build_options(request: &Value, observed_retail_ids: bool) -> ChoiceSet {
    let mut result = ChoiceSet::default();
    if request.get("n").is_some_and(|n| n.as_u64() != Some(1)) {
        result.note(
            "(request)",
            "excluded",
            "Only one requested completion can be replaced",
        );
        return result;
    }
    if let Some(format) = request
        .get("response_format")
        .filter(|value| !value.is_null())
        && format.as_object().is_none_or(|object| {
            object.len() != 1 || object.get("type").and_then(Value::as_str) != Some("text")
        })
    {
        result.note(
            "(request)",
            "excluded",
            "Structured or unknown response formats cannot be replaced by a tool call",
        );
        return result;
    }
    let selected = match request.get("tool_choice") {
        None | Some(Value::Null) => None,
        Some(Value::String(choice)) if choice == "auto" || choice == "required" => None,
        Some(Value::Object(choice))
            if choice.get("type").and_then(Value::as_str) == Some("function") =>
        {
            if choice.len() != 2
                || choice
                    .get("function")
                    .and_then(Value::as_object)
                    .is_none_or(|function| function.len() != 1)
            {
                result.note(
                    "(request)",
                    "excluded",
                    "Unsupported specific tool choice constraints",
                );
                return result;
            }
            match choice
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
            {
                Some(name) if !name.is_empty() => Some(name),
                _ => {
                    result.note("(request)", "excluded", "Malformed specific tool choice");
                    return result;
                }
            }
        }
        _ => {
            result.note(
                "(request)",
                "excluded",
                "Tool choice forbids calls or uses an unsupported policy",
            );
            return result;
        }
    };
    let known = if observed_retail_ids {
        seen_ids(request)
    } else {
        BTreeMap::new()
    };
    let Some(tools) = request.get("tools").and_then(Value::as_array) else {
        result.note("(request)", "excluded", "No tool array available");
        return result;
    };
    let mut counts = BTreeMap::<&str, usize>::new();
    for tool in tools {
        if let Some(name) = tool["function"]["name"].as_str() {
            *counts.entry(name).or_default() += 1;
        }
    }
    for tool in tools {
        let function = &tool["function"];
        let name = function["name"].as_str().unwrap_or("");
        if tool["type"] != "function"
            || name.is_empty()
            || counts.get(name).copied().unwrap_or(0) != 1
        {
            result.note(
                name,
                "excluded",
                "Tool must be a named function with a unique name",
            );
            continue;
        }
        if selected.is_some_and(|selected| selected != name) {
            result.note(name, "excluded", "The request requires a different tool");
            continue;
        }
        match tool_calls(function, &known, observed_retail_ids) {
            Ok((calls, origin)) if result.options.len() + calls.len() <= MAX_OPTIONS => {
                result.note(name, origin, format!("{} complete calls", calls.len()));
                for arguments in calls {
                    result.options.insert(
                        format!("call_{:03}", result.options.len()),
                        CompleteCall {
                            name: name.into(),
                            arguments,
                            origin: origin.into(),
                            description: function["description"]
                                .as_str()
                                .unwrap_or_default()
                                .into(),
                        },
                    );
                }
            }
            Ok(_) => result.note(
                name,
                "excluded",
                "Complete option set exceeds remaining choice capacity; no partial truncation",
            ),
            Err(error) => result.note(name, "excluded", error),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(name: &str, parameters: Value) -> Value {
        json!({"type":"function", "function":{"name":name, "parameters":parameters}})
    }

    fn empty(name: &str) -> Value {
        tool(name, json!({"type":"object","properties":{}}))
    }

    fn unary(name: &str, parameter: &str, schema: Value) -> Value {
        let mut properties = Map::new();
        properties.insert(parameter.into(), schema);
        tool(
            name,
            json!({"type":"object", "properties":properties, "required":[parameter]}),
        )
    }

    fn request(tools: Vec<Value>) -> Value {
        json!({"messages":[], "tools":tools})
    }

    #[test]
    fn complete_calls_cover_optional_omission_bools_enums_and_ranges() {
        let req = request(vec![
            empty("noop"),
            tool(
                "choose",
                json!({"type":"object", "properties":{
            "action":{"type":"string","enum":["return","exchange"]},
            "confirm":{"type":"boolean"},
            "count":{"type":"integer","minimum":1,"maximum":2}}, "required":["action","count"]}),
            ),
        ]);
        let result = build_options(&req, false);
        assert_eq!(result.options.len(), 13);
        assert_eq!(result.options["call_000"].arguments, json!({}));
        assert!(
            result
                .options
                .values()
                .any(|call| call.arguments == json!({"action":"return","count":1}))
        );
        assert!(
            result.options.values().any(
                |call| call.arguments == json!({"action":"exchange","confirm":false,"count":2})
            )
        );
        assert!(result.options.values().all(|call| call.origin == "native"));
        assert_eq!(
            serde_json::to_value(&result).unwrap(),
            serde_json::to_value(build_options(&req, false)).unwrap()
        );
    }

    #[test]
    fn enum_values_must_satisfy_type_numeric_and_string_constraints() {
        let req = request(vec![
            unary(
                "count",
                "n",
                json!({"type":"integer","enum":[true,1,2,3,4,5,6],
                "minimum":1,"maximum":6,"exclusiveMinimum":1,"exclusiveMaximum":6,"multipleOf":2}),
            ),
            unary(
                "word",
                "s",
                json!({"type":"string","enum":["a","éé","three"],"minLength":2,"maxLength":2}),
            ),
        ]);
        let result = build_options(&req, false);
        assert_eq!(result.options.len(), 3);
        assert_eq!(result.options["call_000"].arguments, json!({"n":2}));
        assert_eq!(result.options["call_001"].arguments, json!({"n":4}));
        assert_eq!(result.options["call_002"].arguments, json!({"s":"éé"}));
    }

    #[test]
    fn unknown_or_composite_constraints_do_not_get_ignored() {
        for schema in [
            json!({"type":"string","enum":["a"],"pattern":"^b"}),
            json!({"type":"boolean","const":true}),
            json!({"type":"integer","enum":[1],"oneOf":[{"minimum":2}]}),
            json!({"type":"string","enum":["a"],"format":"email"}),
            json!({"type":"string","enum":["a"],"minLength":-1}),
            json!({"type":"integer","minimum":0,"maximum":3,"multipleOf":0}),
            json!({"type":"integer","minimum":0,"maximum":3,"exclusiveMinimum":true}),
            json!({"type":"integer","enum":[1.5]}),
            json!({"type":"number","enum":[1]}),
            json!({"enum":[null]}),
        ] {
            let result = build_options(&request(vec![unary("bad", "x", schema)]), false);
            assert!(result.options.is_empty(), "{:?}", result);
            assert_eq!(result.audit[0].status, "excluded");
            assert!(!result.audit[0].reason.is_empty());
        }
    }

    #[test]
    fn range_limits_and_optional_products_never_truncate() {
        let huge = unary(
            "huge",
            "n",
            json!({"type":"integer","minimum":i64::MIN,"maximum":i64::MAX}),
        );
        let full = unary(
            "full",
            "n",
            json!({"type":"integer","minimum":0,"maximum":253}),
        );
        assert_eq!(
            build_options(&request(vec![full.clone()]), false)
                .options
                .len(),
            MAX_OPTIONS
        );
        let result = build_options(&request(vec![empty("first"), full, huge]), false);
        assert_eq!(result.options.len(), 1);
        assert!(result.audit[1].reason.contains("no partial truncation"));
        let mut properties = Map::new();
        for i in 0..6 {
            properties.insert(format!("flag_{i}"), json!({"type":"boolean"}));
        }
        let result = build_options(
            &request(vec![tool(
                "optional",
                json!({"type":"object","properties":properties}),
            )]),
            false,
        );
        assert!(result.options.is_empty()); // 3^6, including omission.
    }

    #[test]
    fn tool_policy_and_output_shape_are_honored() {
        let mut req = request(vec![empty("a"), empty("b")]);
        req["tool_choice"] = json!("required");
        assert_eq!(build_options(&req, false).options.len(), 2);
        req["tool_choice"] = json!({"type":"function","function":{"name":"b"}});
        let result = build_options(&req, false);
        assert_eq!(result.options.len(), 1);
        assert_eq!(result.options["call_000"].name, "b");
        req["tool_choice"] = json!("none");
        assert!(build_options(&req, false).options.is_empty());
        req["tool_choice"] = json!("unrecognized");
        assert!(build_options(&req, false).options.is_empty());
        req["tool_choice"] = json!("auto");
        req["n"] = json!(2);
        assert!(build_options(&req, false).options.is_empty());
        req["n"] = json!(1);
        req["response_format"] = json!({"type":"json_object"});
        assert!(build_options(&req, false).options.is_empty());
        req["response_format"] = json!({"type":"text"});
        req["parallel_tool_calls"] = json!(true);
        assert_eq!(build_options(&req, false).options.len(), 2);
    }

    #[test]
    fn malformed_parameter_shapes_and_duplicate_tools_are_excluded() {
        for schema in [
            json!({"type":"object","properties":{},"required":["missing"]}),
            json!({"type":"object","properties":{"a":{"type":"boolean"}},"required":["a","a"]}),
            json!({"type":"object","properties":{},"required":"a"}),
            json!({"type":"object","properties":[],"required":[]}),
            json!({"type":"object","additionalProperties":true}),
            json!({"type":"object","allOf":[{}]}),
            json!({"type":"object","minProperties":1}),
        ] {
            assert!(
                build_options(&request(vec![tool("bad", schema)]), false)
                    .options
                    .is_empty()
            );
        }
        assert!(
            build_options(&request(vec![empty("same"), empty("same")]), false)
                .options
                .is_empty()
        );
    }

    #[test]
    fn retail_adaptation_is_opt_in_and_reads_only_previous_tool_results() {
        let lookup = unary("get_user_details", "user_id", json!({"type":"string"}));
        let mut req = request(vec![lookup]);
        req["messages"] = json!([
            {"role":"user","content":"{\"user_id\":\"do_not_use\"}"},
            {"role":"assistant","tool_calls":[{"id":"lookup1","function":{"name":"find_user_id_by_email","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"lookup1","content":"alice_123"}
        ]);
        assert!(build_options(&req, false).options.is_empty());
        let result = build_options(&req, true);
        assert_eq!(result.options.len(), 1);
        assert_eq!(
            result.options["call_000"].arguments,
            json!({"user_id":"alice_123"})
        );
        assert_eq!(result.options["call_000"].origin, "observed_id_choices");
        req["messages"] = json!([{"role":"tool","tool_call_id":"not_yet_called","content":"future_id"},
            {"role":"assistant","tool_calls":[{"id":"not_yet_called","function":{"name":"find_user_id_by_email"}}]}]);
        assert!(build_options(&req, true).options.is_empty());
    }

    #[test]
    fn retail_identifiers_recover_product_and_order_outputs_without_names() {
        let mut req = request(vec![
            unary(
                "get_product_details",
                "product_id",
                json!({"type":"string"}),
            ),
            unary("get_order_details", "order_id", json!({"type":"string"})),
        ]);
        req["messages"] = json!([
            {"role":"assistant","tool_calls":[{"id":"p1","function":{"name":"list_all_product_types"}}]},
            {"role":"tool","tool_call_id":"p1","content":"{\"Laptop\":\"product_123\"}"},
            {"role":"tool","content":"{\"orders\":[\"#Z\",\"#A\"],\"nested\":[{\"order_id\":\"#Z\"}]}"}
        ]);
        let result = build_options(&req, true);
        assert_eq!(result.options.len(), 3);
        assert_eq!(
            result.options["call_000"].arguments,
            json!({"product_id":"product_123"})
        );
        assert_eq!(
            result.options["call_001"].arguments,
            json!({"order_id":"#A"})
        );
        assert_eq!(
            result.options["call_002"].arguments,
            json!({"order_id":"#Z"})
        );
    }

    #[test]
    fn adaptation_cannot_bypass_schema_constraints_or_errors() {
        let mut req = request(vec![unary(
            "get_user_details",
            "user_id",
            json!({"type":"string","pattern":"^bob"}),
        )]);
        req["messages"] =
            json!([{"role":"tool","name":"find_user_id_by_email","content":"alice_123"}]);
        assert!(build_options(&req, true).options.is_empty());
        req["tools"][0] = tool(
            "get_user_details",
            json!({"type":"object","properties":{"user_id":{"type":"string"}},"required":["missing"]}),
        );
        assert!(build_options(&req, true).options.is_empty());
        req["tools"][0] = unary("get_user_details", "user_id", json!({"type":"string"}));
        for content in ["Error: not found", "{\"error\":\"not found\"}"] {
            req["messages"][0]["content"] = json!(content);
            assert!(build_options(&req, true).options.is_empty());
        }
        req["messages"] = json!([{"role":"tool","name":"find_user_id_by_email","content":"alice_123","is_error":true}]);
        assert!(build_options(&req, true).options.is_empty());
    }
}
