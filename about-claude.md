2. Use /compact proactively — monitor the context meter and run /compact at around 70% capacity, rather than letting it fill up naturally. MCPcat This compresses earlier conversation into a summary, freeing up space without starting fresh.
3. Save session state before clearing — before you hit the limit or start a new session, ask Claude to write a session-notes.md summarising progress, decisions made, and what's next. Then on the next session: /clear → @session-notes.md → continue. This avoids the full re-read.
4. Be surgical with file references — avoid asking Claude to "look at the whole project." Point it at specific files (@src/auth/login.tsx) rather than broad directories. Batch edits: one long diff request burns fewer tokens than several "please refine" follow-ups

commitline
