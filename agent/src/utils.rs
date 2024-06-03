use regex::Regex;

pub fn parse_next_move_and_(
    input: &str,
    next_marker: Option<&str>,
) -> (bool, Option<String>, Vec<String>) {
    let json_regex = Regex::new(r"\{[^}]*\}").unwrap();
    let json_str = json_regex
        .captures(input)
        .and_then(|cap| cap.get(0))
        .map_or(String::new(), |m| m.as_str().to_string());

    let continue_or_terminate_regex =
        Regex::new(r#""continue_or_terminate":\s*"([^"]*)""#).unwrap();
    let continue_or_terminate = continue_or_terminate_regex
        .captures(&json_str)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    let next_move = match next_marker {
        Some(marker) => {
            let next_marker_regex = Regex::new(&format!(r#""{}":\s*"([^"]*)""#, marker)).unwrap();
            Some(
                next_marker_regex
                    .captures(&json_str)
                    .and_then(|cap| cap.get(1))
                    .map_or(String::new(), |m| m.as_str().to_string()),
            )
        }
        None => None,
    };

    let key_points_array_regex = Regex::new(r#""key_points":\s*\[(.*?)\]"#).unwrap();

    let key_points_array = key_points_array_regex
        .captures(&json_str)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    let key_points: Vec<String> = if !key_points_array.is_empty() {
        key_points_array
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .collect()
    } else {
        vec![]
    };

    (&continue_or_terminate == "TERMINATE", next_move, key_points)
}

pub fn parse_planning_steps(input: &str) -> Vec<String> {
    let steps_regex = Regex::new(r#""steps_to_take":\s*(\[[^\]]*\])"#).unwrap();
    let steps_str = steps_regex
        .captures(input)
        .and_then(|cap| cap.get(1))
        .map_or(String::new(), |m| m.as_str().to_string());

    if steps_str.is_empty() {
        eprintln!("Failed to extract 'steps_to_take' from input.");
        return vec![];
    }

    let parsed_steps: Vec<String> = match serde_json::from_str(&steps_str) {
        Ok(val) => val,
        Err(_) => {
            eprintln!("Failed to parse extracted 'steps_to_take' as JSON.");
            return vec![];
        }
    };

    parsed_steps
}
