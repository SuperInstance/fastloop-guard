/// Multi-language syntax validation for sandboxed execution.

/// Validate Python code: block shell metacharacters and injection patterns.
pub fn validate_python(code: &str) -> Result<(), String> {
    // Block dangerous shell metacharacters and patterns
    let blocked = ["$(", "&&", "||", "|", ">>", ">", "<", "`", ";",
                   "__import__", "exec(", "eval(", "os.system", "subprocess"];

    for pattern in blocked {
        if code.contains(pattern) {
            return Err(format!("blocked pattern in Python code: {:?}", pattern));
        }
    }

    // Check balanced parentheses, brackets, braces
    if !balanced_delimiters(code) {
        return Err("unbalanced delimiters in Python code".into());
    }

    Ok(())
}

/// Validate JSON payload by parsing it.
pub fn validate_json(payload: &str) -> Result<(), String> {
    serde_json::from_str::<serde_json::Value>(payload)
        .map(|_| ())
        .map_err(|e| format!("invalid JSON: {}", e))
}

/// Validate Rust code: block unsafe blocks and dangerous std paths.
pub fn validate_rust(code: &str) -> Result<(), String> {
    let blocked = ["unsafe", "std::process::Command", "std::fs::",
                   "std::net::", "std::thread::spawn"];

    for pattern in blocked {
        if code.contains(pattern) {
            return Err(format!("blocked pattern in Rust code: {:?}", pattern));
        }
    }

    if !balanced_delimiters(code) {
        return Err("unbalanced delimiters in Rust code".into());
    }

    Ok(())
}

fn balanced_delimiters(code: &str) -> bool {
    let mut paren = 0i32;
    let mut bracket = 0i32;
    let mut brace = 0i32;
    let mut in_string = false;
    let mut escape_next = false;
    let mut string_char = ' ';

    for ch in code.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if in_string {
            if ch == string_char {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' | '\'' => {
                in_string = true;
                string_char = ch;
            }
            '(' => paren += 1,
            ')' => paren -= 1,
            '[' => bracket += 1,
            ']' => bracket -= 1,
            '{' => brace += 1,
            '}' => brace -= 1,
            _ => {}
        }
        if paren < 0 || bracket < 0 || brace < 0 {
            return false;
        }
    }

    paren == 0 && bracket == 0 && brace == 0 && !in_string
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_valid() {
        assert!(validate_python("x = [1, 2, 3]\nprint(x)").is_ok());
    }

    #[test]
    fn python_shell_injection_blocked() {
        assert!(validate_python("import os; os.system('rm -rf /')").is_err());
    }

    #[test]
    fn python_subprocess_blocked() {
        assert!(validate_python("subprocess.run(['ls'])").is_err());
    }

    #[test]
    fn python_pipe_blocked() {
        assert!(validate_python("x = 'a' | 'b'").is_err());
    }

    #[test]
    fn python_unbalanced_parens() {
        assert!(validate_python("foo(bar(").is_err());
    }

    #[test]
    fn json_valid() {
        assert!(validate_json(r#"{"key": "value"}"#).is_ok());
    }

    #[test]
    fn json_truncated_fails() {
        assert!(validate_json(r#"{"key": "val"#).is_err());
    }

    #[test]
    fn json_array_valid() {
        assert!(validate_json("[1, 2, 3]").is_ok());
    }

    #[test]
    fn rust_valid() {
        assert!(validate_rust("fn main() { let x = 42; }").is_ok());
    }

    #[test]
    fn rust_unsafe_blocked() {
        assert!(validate_rust("unsafe { *ptr }").is_err());
    }

    #[test]
    fn rust_std_fs_blocked() {
        assert!(validate_rust("std::fs::read_to_string(\"/etc/passwd\")").is_err());
    }

    #[test]
    fn rust_std_process_blocked() {
        assert!(validate_rust("std::process::Command::new(\"ls\")").is_err());
    }

    #[test]
    fn rust_unbalanced_braces() {
        assert!(validate_rust("fn main() {").is_err());
    }

    #[test]
    fn balanced_with_strings() {
        assert!(balanced_delimiters("x = \"(hello)\""));
    }
}
