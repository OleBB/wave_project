2. Use /compact proactively — monitor the context meter and run /compact at around 70% capacity, rather than letting it fill up naturally. MCPcat This compresses earlier conversation into a summary, freeing up space without starting fresh.
3. Save session state before clearing — before you hit the limit or start a new session, ask Claude to write a session-notes.md summarising progress, decisions made, and what's next. Then on the next session: /clear → @session-notes.md → continue. This avoids the full re-read.
4. Be surgical with file references — avoid asking Claude to "look at the whole project." Point it at specific files (@src/auth/login.tsx) rather than broad directories. Batch edits: one long diff request burns fewer tokens than several "please refine" follow-ups

Then next session just open the project and the first thing Claude sees is section 0 with the exact run, the numbers, and the three next steps. No `/clear`, no re-explaining.

The `session-notes.md` approach in your `about-claude.md` is useful when you've been going deep into code changes mid-session and want to freeze state. Here you've been proactive enough that `claude.md` itself serves that role — it's persistent, version-controlled, and purpose-built for this project.
