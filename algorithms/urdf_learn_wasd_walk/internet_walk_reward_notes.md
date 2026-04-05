# Internet Notes: Walking / Running Reward Design

Last updated: 2026-04-04

This note records the external references I used to redesign the Landau Stage A locomotion task.

## Sources

1. Isaac Lab reward API docs
   - <https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.envs.mdp.html>
   - Isaac Lab exposes generic building blocks such as `track_lin_vel_xy_exp`, `desired_contacts`, `undesired_contacts`, and `contact_forces`.
   - Takeaway for this repo: stock velocity tracking plus generic contact penalties are necessary but not sufficient. They do not define a natural gait on their own.

2. Wu et al., "Infer and Adapt: Bipedal Locomotion Reward Learning from Demonstrations via Inverse Reinforcement Learning" (ICRA 2024)
   - <https://lab-idar.gatech.edu/wp-content/uploads/2023/09/ICRA2024_IRL_Reward_Shaping_Wu.pdf>
   - Main takeaway: reward functions inferred from demonstrations can encode locomotion knowledge that transfers better to unseen settings than purely hand-tuned rewards.
   - Practical inference for this repo: without demonstrations, we should still imitate the same discipline by adding diagnostics that reflect expert gait structure, not just speed.

3. Kumar et al., "Learning Goal-Following Locomotion Controllers for Humanoids Using Demonstration and Reinforcement Learning"
   - <https://openreview.net/pdf?id=r0xwZWjLEi>
   - The paper uses imitation pretraining to get humanlike gait dynamics and coordinated arm-leg motion, then fine-tunes with RL.
   - Practical inference for this repo: if we want truly humanlike gait later, motion priors or imitation pretraining will likely matter. Pure RL can be improved a lot, but it will still tend to exploit loopholes unless we explicitly regularize gait structure.

4. "Statistical Reward Shaping for Reinforcement Learning in Bipedal Locomotion" (MDPI Electronics, 2026)
   - <https://www.mdpi.com/2079-9292/15/6/1203>
   - Useful points:
     - reward design should be monitored using more than forward distance alone
     - the reward stack should cover task, gait pattern, posture/stability, and energy/smoothness
     - excessive foot-slide penalty can hurt posture and natural gait instead of helping
   - Practical inference for this repo: keep slide penalties moderate and evaluate with multi-metric gait diagnostics, not speed only.

5. "Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion" (arXiv 2025)
   - <https://ar5iv.labs.arxiv.org/html/2505.20619>
   - Useful points:
     - walking rewards should emphasize contact pattern, foot clearance, and support posture
     - running rewards should emphasize push-off, flight phase, and short contact duration
     - mixing walk and run rewards too early creates reward conflict; staged curricula help
   - Practical inference for this repo: Stage A should focus on proper walking / jog structure first. True running should be a later mode or a separate reward slice.

6. Siekmann et al., "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition" (2021)
   - <https://arxiv.org/abs/2011.01387>
   - Useful points:
     - gait quality improves when the reward names specific stance and swing behaviors instead of only rewarding net motion
     - phase-structured rewards can separate walking from running by contact timing and foot force / velocity structure
   - Practical inference for this repo: if the current event-based shaping still converges to a shuffle, the next escalation should be a light gait-phase prior instead of piling on more generic dense penalties.

7. Balasubramanian et al., "Foot placement in a body reference frame during walking and its relationship to step length" (Clinical Biomechanics, 2010)
   - <https://pmc.ncbi.nlm.nih.gov/articles/PMC2881577/>
   - Useful points:
     - step geometry measured relative to the pelvis / body frame captures control structure that plain foot-to-foot spacing misses
     - asymmetry is easier to diagnose in a body frame than in a world frame
   - Practical inference for this repo: touchdown should be judged relative to `root_x` or an equivalent body frame, not only by global displacement or foot travel.

8. Bruijn and van Dieen, "Control of human gait stability through foot placement" (Journal of the Royal Society Interface, 2018)
   - <https://pmc.ncbi.nlm.nih.gov/articles/PMC6030625/>
   - Useful points:
     - stable walking depends strongly on where the foot lands relative to body state, especially in the mediolateral direction
     - foot-placement control is a first-class stability mechanism, not a cosmetic style feature
   - Practical inference for this repo: validator and rewards should care about touchdown placement and stance width together, because wide-foot shuffles are often a stability hack.

9. Vu et al., "A Review of Gait Phase Detection Algorithms for Lower Limb Prostheses" (Sensors, 2020)
   - <https://pmc.ncbi.nlm.nih.gov/articles/PMC7411778/>
   - Useful points:
     - normal walking is organized around stance, swing, and two double-support intervals
     - contact timing is therefore part of the gait definition, not just a side effect of speed
   - Practical inference for this repo: Stage A walk validation should inspect single-support ratio, double-support presence, and low flight ratio together rather than using one timing scalar.

10. Bergamini et al., "Walking symmetry is speed and index dependent" (Scientific Reports, 2024)
    - <https://pmc.ncbi.nlm.nih.gov/articles/PMC11341956/>
    - Useful points:
      - healthy walking is not perfectly symmetric, and the acceptable asymmetry band changes with walking speed
      - symmetry metrics need context; over-constraining to 50/50 can reject valid walking
    - Practical inference for this repo: validator symmetry checks should be bounded, not exact. Large one-sided stepping is bad, but mild left/right mismatch should be tolerated.

11. Gregg et al., "On the mechanics of functional asymmetry in bipedal walking" (PLoS ONE, 2014)
    - <https://pmc.ncbi.nlm.nih.gov/articles/PMC4201655/>
    - Useful points:
      - mechanically stable asymmetric gait families can emerge even in symmetric bipeds
      - symmetry alone cannot define normality without considering speed and support structure
    - Practical inference for this repo: asymmetry should be a warning signal, not the only acceptance criterion.

12. Minimum toe-clearance review material
    - <https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2014.00243/pdf>
    - Useful points:
      - mid-swing toe clearance is a critical anti-trip event
      - low or inconsistent clearance is a concrete signature of dragging or non-committal swing
    - Practical inference for this repo: swing clearance belongs in the validator even if the exact threshold is robot-specific.

13. Fukuchi et al., "Mechanics of very slow human walking" (2019)
    - <https://pmc.ncbi.nlm.nih.gov/articles/PMC6889403/>
    - Useful points:
      - as walking speed decreases, step length decreases and double-support time increases
      - step width itself does not have to grow just because gait gets slower
    - Practical inference for this repo: a slow walk is not an excuse for a wide split-stance. Stage A should tolerate slower commanded walking, but it should not accept speed loss that is bought by widening the base.

14. Sanz-Merodio et al., "Biped Gait Stability Classification Based on the Predicted Step Viability" (2024)
    - <https://pmc.ncbi.nlm.nih.gov/articles/PMC11118085/>
    - Useful points:
      - stable gait is not only about the current contact; it is about whether the current step leaves a viable next step
      - capture-point / step-viability ideas formalize why foot placement relative to body state matters
    - Practical inference for this repo: touchdown geometry should be judged by whether the root can move past the current base, not only by whether one foot briefly lands ahead in world space.

## What Counts As Proper Walking Here

For this repo, "proper walking" should mean all of the following, not just one:

1. The robot moves primarily along the commanded forward axis.
2. Non-support links do not create the motion.
3. The gait contains real left/right alternation with meaningful single-support time and some double-support structure.
4. The unloaded side lifts above the stance side instead of dragging.
5. Touchdowns place the landing side ahead of the opposite support side in the body frame, not behind it.
6. Touchdowns keep the root/body state between the stance and landing feet often enough that the robot is really stepping past its base, not just widening it.
7. Average support width stays near nominal instead of widening into a stabilizing split stance.
8. Torso/control-root height stays upright enough that the robot is not effectively seated or crouch-dragging.
9. Action smoothness and slip penalties remain present, but not so strong that they suppress gait formation.

If one of these fails, the motion may still look "successful" under naive forward-distance metrics while actually being a cheating solution.

## Walking vs Running

The references strongly suggest that walking and running should not share exactly the same gait shaping:

- Walking:
  - alternating support timing
  - foot clearance
  - stance posture
  - moderate support width
  - stable double/single-support transitions

- Running:
  - stronger push-off
  - shorter contact duration
  - explicit flight phase
  - larger commanded speed band
  - possibly different contact-pattern rewards than walking

Current decision for this repo:

- Stage A remains a proper walking / jog task.
- I am not treating Stage A as the place to solve true fast running.
- If later work targets real running, it should use speed-conditioned rewards or a separate stage rather than stretching the walk reward until it breaks.

Practical walk vs run definition for this repo:

- Walking:
  - low flight ratio
  - repeated single-support windows on both sides
  - visible double-support transitions
  - touchdown placement ahead of the opposite side in the body frame
  - moderate stance width

- Running:
  - explicit aerial phases
  - shorter contact durations / lower duty factor
  - stronger push-off and less double support
  - likely a different speed band and reward slice than Stage A

Important caveat from the literature:

- duty factor and contact timing matter, but they are not enough by themselves
- a grounded shuffle can still fake "walking-like" timing
- that is why this repo now combines contact timing, touchdown geometry, swing clearance, support width, and anti-crawl checks

## Repo Actions Taken From These Notes

Actions already reflected in local code on 2026-04-04:

- tightened grouped contact logic so whole-side airborne events matter more than heel/toe rocking
- added deadbanded grouped single-support timing instead of rewarding tiny unload blips
- added touchdown-ahead shaping so a landing foot must meaningfully step ahead of the opposite side
- added swing-height-difference shaping so unloaded support bundles must lift
- tightened stance-width shaping and validator checks
- added stricter validator metrics for single-support ratio, swing clearance, touchdown step length, and support width
- added gentle fine-tune controls for PPO resume runs instead of using fresh-run update aggressiveness

Additional actions added later on 2026-04-04:

- added touchdown-event penalties for step-length deficit so "landing behind" is explicitly bad instead of merely unrewarded
- added touchdown-event penalties for overly wide landings so the robot is punished for split-stance touchdown geometry without freezing the whole episode under a dense width penalty
- kept root-height and anti-crawl shaping, but reduced the root-centered gait terms after they started to collapse fine-tunes into slow constrained shuffles
- aligned touchdown and support geometry around full `foot + toe` support bundles instead of primary feet only, because the validator and the real support polygon both depend on the full bundle
- added a prolonged-double-support penalty after the literature review and local runs showed the same failure mode from the other direction:
  - pure anti-flight shaping pushed the policy away from aerial cheating
  - but without an explicit cost on long double support it could still cash out the reward as a wide, slow shuffle

Latest empirical conclusion from the 2026-04-04 internet-guided redesign pass:

- the repo now has a materially better definition of "proper walking" than it had before
- but the latest training runs still did not produce a dramatic strict-pass checkpoint
- the best new run improved some anti-cheat metrics such as root height and non-support contacts, yet still failed the strict walk bar on width, flight ratio, and yaw drift
- that result supports the literature-based conclusion that scalar reward tuning is nearing its limit here
- the next serious improvement path is likely one of:
  - explicit gait-phase reward routing with target stance/swing/double-support windows
  - a viability / capture-point style touchdown objective
  - imitation or motion-prior guidance

Current working rule from the internet pass:

- positive shaping should teach real stepping
- validator and penalties should police cheating support geometry
- if that still converges to a shuffle, the next move is not "more generic penalties"; it is a phase-structured gait prior or imitation-style guidance

## Not Implemented Yet

The literature also points to two stronger ideas that are probably worth future work:

1. Motion priors or imitation
   - best route if the target is genuinely humanlike gait rather than just non-cheating locomotion

2. Gait-conditioned reward routing
   - likely the cleanest route once the repo needs one policy that can stand, walk, jog, and run without reward conflict
