# How to Run the Bot from GitHub (No Laptop Needed)

## Step 1 — Register your agent (one-time, ~5 min)

Do this in the Claude chat while your internet is on.

Ask Claude: "Read https://arena.dev.fun/skills/arena.md and help me register"

It will:
- Read the Arena onboarding guide
- Propose a name + bio for your agent
- Register you and save a `.arena-credentials` file in the `examples/` folder
- Give you a claim URL to share

## Step 2 — Push code to GitHub

1. Go to github.com → New repository → name it `poker-agent` → private
2. In Terminal (Poker folder):
   ```bash
   git init
   git add .
   git commit -m "BlackRain79 competition agent"
   git remote add origin https://github.com/YOUR_USERNAME/poker-agent.git
   git push -u origin main
   ```

## Step 3 — Add secrets to GitHub

Go to your repo → Settings → Secrets and variables → Actions → New repository secret

Add these secrets:

| Secret name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key (sk-ant-...) |
| `ARENA_COMPETITION_ID` | `seed_poker_eval_s1` (or your competition ID) |
| `ARENA_CREDENTIALS` | Contents of `examples/.arena-credentials` (after registering) |
| `ARENA_API_KEY` | Your Arena API key (from `.arena-credentials` file, the `apiKey` field) |

## Step 4 — Run the bot from GitHub

1. Go to your repo → Actions tab → "Run Poker Competition Bot"
2. Click "Run workflow"
3. Choose mode: `llm` (Claude, best) or `heuristic` (free)
4. Click green "Run workflow" button
5. Close your laptop — the bot runs on GitHub's servers

The full 500-hand match takes 30–45 min. You can watch the logs live in the Actions tab.

## Step 5 — Check your score

Go to https://b-arena.dev.fun/poker-eval and check the leaderboard.
