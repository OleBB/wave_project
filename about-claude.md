2. Use /compact proactively — monitor the context meter and run /compact at around 70% capacity, rather than letting it fill up naturally. MCPcat This compresses earlier conversation into a summary, freeing up space without starting fresh.
3. Save session state before clearing — before you hit the limit or start a new session, ask Claude to write a session-notes.md summarising progress, decisions made, and what's next. Then on the next session: /clear → @session-notes.md → continue. This avoids the full re-read.
4. Be surgical with file references — avoid asking Claude to "look at the whole project." Point it at specific files (@src/auth/login.tsx) rather than broad directories. Batch edits: one long diff request burns fewer tokens than several "please refine" follow-ups

Then next session just open the project and the first thing Claude sees is section 0 with the exact run, the numbers, and the three next steps. No `/clear`, no re-explaining.

The `session-notes.md` approach in your `about-claude.md` is useful when you've been going deep into code changes mid-session and want to freeze state. Here you've been proactive enough that `claude.md` itself serves that role — it's persistent, version-controlled, and purpose-built for this project.


#min lange gode oppsummering til claude:

And: work on the overall presentation structure for the thesis. This will be outlined in main save_figure.py. 


Logically, kind of in the same manner as main.py works.

Chapter 04 - Methodology:
Steps: 1 check the probe error. 2 check the stillwater timing (how long to wait between sessions (longer waves create more low frequency swell. (but wind changes this game alot, little waiting time really required)). 3 probe-placeement longitudinal/lateral and its effects vs panels or wind. 4 Wind, more subtopics(?). 5 how does a full signal look. 6 how do we find the wave-range for the selected wavetrain? Threshold, rampup and stable train. 7. Auto-correlation-A: how stable is our wavetrain. 8 autocorrelation-B: How equal is a wave laterally wind vs nowind (discovered from parallel probes). 9 ... (possibly more steps as mentioned in processor or processor2nd but i cannot remember) 

Then Chapter 05 - Results.
... damping. damping vs freq (this is the meaningful one). damping vs amplitude (less happening, but thats a finding too).
And that damping based on WindCondition. Note that this is the single key question: "How does the wind affect the damping?" That is it. It can be expanded to "How much is left of the main mode(freq) of the wave travelling thru the geometry (consisting of many elastic panels), when wind is added?". 


and last note: ak, or ka . wavenumber and wavesteepness - its a calculation we need to make sure follows along all our plot, as one of the Keys. 

Lets remember the keys (save for an overarching understanding): INPUT KEYs: Amp [Voltage], Freq, PanelCondition, WindCondition. OUTPUT KEYs: OUT/IN (FFT) = damping, witout being tricked by the steep but short wind waves. ka = found(not pre-calculated, but truly found per probe)

ok, save for later: ka debate - what constitutes our wave? The wave as seen right before the panellocation, at 9373=IN, but without panel(no reflection - a "perfect world"). This would be the base wave expected to be delivered from the wavemaker. But reality is uglier... the panel dampens the shorter waves alot. this means the IN and the OUT has different amplitude and ak. but ok the frequency of that exact wave does not change by far as much as the amplitude does. Amplitude can be damped ive seen up to 95%, but i still can follow the same wave, because the base frequency is still mostly there.
