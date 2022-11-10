register_user_schema = {
    "type": "object",
    "properties": {
        "username": {"type": "string"},
    },
    "required": ["username"]
}

generate_token_schema = {
    "type": "object",
    "properties": {
        "username": {"type": "string"},
    },
    "required": ["username"]
}
query_schema = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
    },
    "required": ["prompt"]
}