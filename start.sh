#!/bin/bash
# Write credentials from environment variable
echo "{\"apiKey\":\"$ARENA_API_KEY\",\"agentId\":\"cmpz7ybxl0djqdfirzvedc2ea\",\"handle\":\"adah_rain\",\"name\":\"Adah Rain\"}" > examples/.arena-credentials
echo "Credentials written for Adah Rain"

# Run the playground bot
cd examples && python3 playground_agent.py
