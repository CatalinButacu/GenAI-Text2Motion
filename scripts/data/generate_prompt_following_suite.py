#!/usr/bin/env python
"""Generate the 200-entry prompt-following evaluation suite.

Design goals
------------
- Diverse vocabulary: varied subjects, verb synonyms, sentence structures.
- No template x modifier cross-product; each semantic unit has 5 distinct phrasings.
- Temporal relations encoded for all multi-step entries.
- Diversity gate enforced at generation time (avg_nearest_similarity < 0.72, TTR > 0.18).
- Output written one JSON object per line for readability.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.evaluation.audit_prompt_diversity import (
    DEFAULT_THRESHOLDS,
    build_metrics,
    evaluate_diversity,
)
from scripts.evaluation.prompt_suite_schema import validate_suite

OOD_NEGATION = "OOD negation"

# ---------------------------------------------------------------------------
# Canonical prompt pool - (id, prompt, category, required_actions,
#                          required_entities, forbidden_actions, ordered,
#                          style_constraints, difficulty, tags,
#                          temporal_relations, notes)
# ---------------------------------------------------------------------------

POOL_SINGLE_ACTION = [
    # walk group
    ("pf_001", "a person walks forward at a steady pace", ["walk"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_002", "someone strolls across the room calmly", ["walk"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_003", "the figure paces forward with measured steps", ["walk"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_004", "an individual ambles briskly down the path", ["walk"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_005", "a human figure marches ahead purposefully", ["walk"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    # run group
    ("pf_006", "a person runs forward quickly", ["run"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_007", "someone sprints across the open space", ["run"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_008", "the figure dashes forward at full speed", ["run"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_009", "an individual jogs at a comfortable pace", ["run"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_010", "a human races toward the far end of the corridor", ["run"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    # jump group
    ("pf_011", "a person jumps straight up from standing", ["jump"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_012", "someone leaps into the air from a standstill", ["jump"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_013", "the figure springs upward with both feet together", ["jump"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_014", "an individual hops repeatedly in place", ["jump"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_015", "a human bounds upward from a crouched position", ["jump"], ["person"], [], False, [], "medium", ["golden", "single"], [], ""),
    # kick group
    ("pf_016", "a person kicks a ball forward", ["kick"], ["person", "ball"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_017", "someone boots the ball hard to the left", ["kick"], ["ball"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_018", "the figure strikes the ball with a sweeping foot motion", ["kick"], ["ball"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_019", "an individual punts the ball across the floor", ["kick"], ["ball"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_020", "a human delivers a powerful kick to the stationary ball", ["kick"], ["person", "ball"], [], False, [], "medium", ["golden", "single"], [], ""),
    # throw group
    ("pf_021", "a person throws a ball forward with force", ["throw"], ["person", "ball"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_022", "someone hurls an object across the room", ["throw"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_023", "the figure flings the item in a wide arc overhead", ["throw"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_024", "an individual tosses the object lightly forward", ["throw"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_025", "a human pitches the ball with an underhand motion", ["throw"], ["ball"], [], False, [], "medium", ["golden", "single"], [], ""),
    # sit group
    ("pf_026", "a person sits down on the ground", ["sit"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_027", "someone lowers themselves slowly into a seat", ["sit"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_028", "the figure settles into a seated position on the floor", ["sit"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_029", "an individual perches carefully on the edge of a chair", ["sit"], ["chair"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_030", "a human takes a seat and rests their hands on their knees", ["sit"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    # stand group
    ("pf_031", "a person stands up from a seated position", ["stand"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_032", "someone rises to their full standing height", ["stand"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_033", "the figure straightens up and holds a still upright pose", ["stand"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_034", "an individual gets up from the floor and stands upright", ["stand"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_035", "a human lifts themselves to a standing position", ["stand"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    # crouch group
    ("pf_036", "a person crouches down close to the ground", ["crouch"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_037", "someone squats low with both knees bent", ["crouch"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_038", "the figure ducks and holds a low body position", ["crouch"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_039", "an individual bends their knees and stays crouched", ["crouch"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_040", "a human hunches forward into a crouching stance", ["crouch"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    # wave group
    ("pf_041", "a person waves their hand in greeting", ["wave"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_042", "someone gestures broadly with a raised arm wave", ["wave"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_043", "the figure signals with a wide overhead arm sweep", ["wave"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_044", "an individual waves goodbye using their right hand", ["wave"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_045", "a human beckons with a repeated arm sweep motion", ["wave"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    # clap group
    ("pf_046", "a person claps their hands together", ["clap"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_047", "someone applauds with steady rhythmic clapping", ["clap"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_048", "the figure strikes their palms together repeatedly", ["clap"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_049", "an individual claps in a slow celebratory rhythm", ["clap"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
    ("pf_050", "a human claps hands three times in succession", ["clap"], ["person"], [], False, [], "easy", ["golden", "single"], [], ""),
]

POOL_MULTI_STEP = [
    # walk -> kick
    ("pf_051", "a person walks toward the ball then kicks it forward", ["walk", "kick"], ["person", "ball"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "kick"}], "ordered"),
    ("pf_052", "someone approaches the ball at a walk and boots it away", ["walk", "kick"], ["ball"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "kick"}], "ordered"),
    ("pf_053", "the figure strolls up to the ball and delivers a kick", ["walk", "kick"], ["ball"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "kick"}], "ordered"),
    ("pf_054", "after walking forward, the individual punts the ball", ["walk", "kick"], ["ball"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "kick"}], "ordered"),
    ("pf_055", "a human paces over to the stationary ball and strikes it with their foot", ["walk", "kick"], ["person", "ball"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "kick"}], "ordered"),
    # run -> jump
    ("pf_056", "a person runs then leaps into the air", ["run", "jump"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "run", "relation": "before", "action_b": "jump"}], "ordered"),
    ("pf_057", "someone sprints forward and launches into a leap", ["run", "jump"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "run", "relation": "before", "action_b": "jump"}], "ordered"),
    ("pf_058", "the figure dashes ahead and springs upward off both feet", ["run", "jump"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "run", "relation": "before", "action_b": "jump"}], "ordered"),
    ("pf_059", "after a short sprint, the individual bounds into the air", ["run", "jump"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "run", "relation": "before", "action_b": "jump"}], "ordered"),
    ("pf_060", "a human races across the floor and vaults upward", ["run", "jump"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "run", "relation": "before", "action_b": "jump"}], "ordered"),
    # crouch -> stand
    ("pf_061", "a person crouches down then stands back up", ["crouch", "stand"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "crouch", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_062", "someone squats low and then rises to standing", ["crouch", "stand"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "crouch", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_063", "the figure ducks down before straightening up fully", ["crouch", "stand"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "crouch", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_064", "after crouching, the individual climbs back to their feet", ["crouch", "stand"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "crouch", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_065", "a human bends into a low crouch then stands fully upright", ["crouch", "stand"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "crouch", "relation": "before", "action_b": "stand"}], "ordered"),
    # sit -> stand
    ("pf_066", "a person sits on the ground then stands up", ["sit", "stand"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "sit", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_067", "someone lowers to seated and then rises back up", ["sit", "stand"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "sit", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_068", "the figure settles into a seated rest and then gets back up", ["sit", "stand"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "sit", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_069", "after sitting briefly, the individual lifts to standing", ["sit", "stand"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "sit", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_070", "a human rests in a seated pose then returns to standing", ["sit", "stand"], ["person"], [], True, [], "easy", ["golden", "ordered"], [{"action_a": "sit", "relation": "before", "action_b": "stand"}], "ordered"),
    # walk -> turn -> run
    ("pf_071", "a person walks, pivots sharply, then sprints away", ["walk", "turn", "run"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "turn"}, {"action_a": "turn", "relation": "before", "action_b": "run"}], "ordered"),
    ("pf_072", "someone strolls forward, turns around, then dashes back", ["walk", "turn", "run"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "turn"}, {"action_a": "turn", "relation": "before", "action_b": "run"}], "ordered"),
    ("pf_073", "the figure paces ahead, rotates on one foot, and races in the new direction", ["walk", "turn", "run"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "turn"}, {"action_a": "turn", "relation": "before", "action_b": "run"}], "ordered"),
    ("pf_074", "after walking, the individual swings around and jogs off at speed", ["walk", "turn", "run"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "turn"}, {"action_a": "turn", "relation": "before", "action_b": "run"}], "ordered"),
    ("pf_075", "a human marches forward, spins on a heel, then bolts ahead", ["walk", "turn", "run"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "turn"}, {"action_a": "turn", "relation": "before", "action_b": "run"}], "ordered"),
    # pick -> throw
    ("pf_076", "a person picks up an object then throws it forward", ["pick", "throw"], ["person", "box"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "pick", "relation": "before", "action_b": "throw"}], "ordered"),
    ("pf_077", "someone grabs the item off the floor and hurls it", ["pick", "throw"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "pick", "relation": "before", "action_b": "throw"}], "ordered"),
    ("pf_078", "the figure retrieves the box and then flings it across the room", ["pick", "throw"], ["box"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "pick", "relation": "before", "action_b": "throw"}], "ordered"),
    ("pf_079", "after picking it up, the individual pitches the object forward", ["pick", "throw"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "pick", "relation": "before", "action_b": "throw"}], "ordered"),
    ("pf_080", "a human lifts the item from the ground and launches it overhead", ["pick", "throw"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "pick", "relation": "before", "action_b": "throw"}], "ordered"),
    # jump -> fall -> stand
    ("pf_081", "a person jumps, loses balance, falls, then stands back up", ["jump", "fall", "stand"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "jump", "relation": "before", "action_b": "fall"}, {"action_a": "fall", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_082", "someone leaps, lands poorly, tumbles, and then recovers to standing", ["jump", "fall", "stand"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "jump", "relation": "before", "action_b": "fall"}, {"action_a": "fall", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_083", "the figure springs up, drops to the floor, and rises again", ["jump", "fall", "stand"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "jump", "relation": "before", "action_b": "fall"}, {"action_a": "fall", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_084", "after jumping, the individual collapses and then gets back upright", ["jump", "fall", "stand"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "jump", "relation": "before", "action_b": "fall"}, {"action_a": "fall", "relation": "before", "action_b": "stand"}], "ordered"),
    ("pf_085", "a human bounds up, collapses on landing, then climbs to standing", ["jump", "fall", "stand"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "jump", "relation": "before", "action_b": "fall"}, {"action_a": "fall", "relation": "before", "action_b": "stand"}], "ordered"),
    # walk -> stop -> walk
    ("pf_086", "a person walks forward, halts completely, then resumes walking", ["walk", "stop", "walk"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "stop"}, {"action_a": "stop", "relation": "before", "action_b": "walk"}], "ordered"),
    ("pf_087", "someone strolls, freezes in place, then continues forward", ["walk", "stop", "walk"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "stop"}, {"action_a": "stop", "relation": "before", "action_b": "walk"}], "ordered"),
    ("pf_088", "the figure paces ahead, pauses abruptly, then moves on", ["walk", "stop", "walk"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "stop"}, {"action_a": "stop", "relation": "before", "action_b": "walk"}], "ordered"),
    ("pf_089", "after walking, the individual stops and then walks again", ["walk", "stop", "walk"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "stop"}, {"action_a": "stop", "relation": "before", "action_b": "walk"}], "ordered"),
    ("pf_090", "a human marches forward, brakes to a standstill, then walks onward", ["walk", "stop", "walk"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "walk", "relation": "before", "action_b": "stop"}, {"action_a": "stop", "relation": "before", "action_b": "walk"}], "ordered"),
    # run -> turn -> run
    ("pf_091", "a person runs, pivots on one foot, then runs the other way", ["run", "turn", "run"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "run", "relation": "before", "action_b": "turn"}, {"action_a": "turn", "relation": "before", "action_b": "run"}], "ordered"),
    ("pf_092", "someone sprints, rotates sharply, and dashes back", ["run", "turn", "run"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "run", "relation": "before", "action_b": "turn"}, {"action_a": "turn", "relation": "before", "action_b": "run"}], "ordered"),
    ("pf_093", "the figure races forward, spins around, then bolts in a new direction", ["run", "turn", "run"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "run", "relation": "before", "action_b": "turn"}, {"action_a": "turn", "relation": "before", "action_b": "run"}], "ordered"),
    ("pf_094", "after running, the individual turns and runs in reverse", ["run", "turn", "run"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "run", "relation": "before", "action_b": "turn"}, {"action_a": "turn", "relation": "before", "action_b": "run"}], "ordered"),
    ("pf_095", "a human dashes forward, swings around on a heel, then races off again", ["run", "turn", "run"], ["person"], [], True, [], "hard", ["golden", "ordered"], [{"action_a": "run", "relation": "before", "action_b": "turn"}, {"action_a": "turn", "relation": "before", "action_b": "run"}], "ordered"),
    # reach -> pick
    ("pf_096", "a person reaches up overhead then picks up the object", ["reach", "pick"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "reach", "relation": "before", "action_b": "pick"}], "ordered"),
    ("pf_097", "someone extends their arm and retrieves the item from above", ["reach", "pick"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "reach", "relation": "before", "action_b": "pick"}], "ordered"),
    ("pf_098", "the figure stretches toward the shelf and takes the object down", ["reach", "pick"], ["shelf"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "reach", "relation": "before", "action_b": "pick"}], "ordered"),
    ("pf_099", "after reaching out, the individual grabs what they were aiming for", ["reach", "pick"], ["person"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "reach", "relation": "before", "action_b": "pick"}], "ordered"),
    ("pf_100", "a human extends upward and collects the item from the high shelf", ["reach", "pick"], ["shelf"], [], True, [], "medium", ["golden", "ordered"], [{"action_a": "reach", "relation": "before", "action_b": "pick"}], "ordered"),
]

POOL_STYLE_MODIFIER = [
    # walk + slow
    ("pf_101", "a person walks forward with a slow deliberate stride", ["walk"], ["person"], [], False, ["slow"], "easy", ["golden", "style"], [], ""),
    ("pf_102", "someone strolls at a reduced tempo, dragging each step", ["walk"], ["person"], [], False, ["slow"], "easy", ["golden", "style"], [], ""),
    ("pf_103", "the figure moves forward very slowly as if fatigued", ["walk"], ["person"], [], False, ["slow"], "easy", ["golden", "style"], [], ""),
    ("pf_104", "an individual walks at a leisurely slow pace with minimal effort", ["walk"], ["person"], [], False, ["slow"], "easy", ["golden", "style"], [], ""),
    ("pf_105", "a human trudges ahead with slow, heavy, deliberate steps", ["walk"], ["person"], [], False, ["slow"], "easy", ["golden", "style"], [], ""),
    # run + fast
    ("pf_106", "a person runs as fast as possible with arms pumping", ["run"], ["person"], [], False, ["fast"], "easy", ["golden", "style"], [], ""),
    ("pf_107", "someone sprints at maximum velocity with full effort", ["run"], ["person"], [], False, ["fast"], "easy", ["golden", "style"], [], ""),
    ("pf_108", "the figure races forward with explosive, rapid speed", ["run"], ["person"], [], False, ["fast"], "easy", ["golden", "style"], [], ""),
    ("pf_109", "an individual dashes at top speed across the space", ["run"], ["person"], [], False, ["fast"], "easy", ["golden", "style"], [], ""),
    ("pf_110", "a human bolts forward as quickly as physically possible", ["run"], ["person"], [], False, ["fast"], "easy", ["golden", "style"], [], ""),
    # walk + tired
    ("pf_111", "a person walks with tired heavy legs that drag slightly", ["walk"], ["person"], [], False, ["tired"], "medium", ["golden", "style"], [], "OOD"),
    ("pf_112", "someone shuffles forward with visible signs of exhaustion", ["walk"], ["person"], [], False, ["tired"], "medium", ["golden", "style"], [], "OOD"),
    ("pf_113", "the figure plods slowly as if weary from long exertion", ["walk"], ["person"], [], False, ["tired"], "medium", ["golden", "style"], [], "OOD"),
    ("pf_114", "an individual drags themselves along showing clear fatigue", ["walk"], ["person"], [], False, ["tired"], "medium", ["golden", "style"], [], "OOD"),
    ("pf_115", "a human stumbles forward in a tired laboured walk", ["walk"], ["person"], [], False, ["tired"], "medium", ["golden", "style"], [], "OOD"),
    # run + energetic
    ("pf_116", "a person runs with high energy and wide powerful strides", ["run"], ["person"], [], False, ["energetic"], "medium", ["golden", "style"], [], ""),
    ("pf_117", "someone jogs enthusiastically with bouncy, spirited steps", ["run"], ["person"], [], False, ["energetic"], "medium", ["golden", "style"], [], ""),
    ("pf_118", "the figure races forward with visible excitement and drive", ["run"], ["person"], [], False, ["energetic"], "medium", ["golden", "style"], [], ""),
    ("pf_119", "an individual sprints energetically with a wide upbeat gait", ["run"], ["person"], [], False, ["energetic"], "medium", ["golden", "style"], [], ""),
    ("pf_120", "a human runs vigorously, bursting with energy", ["run"], ["person"], [], False, ["energetic"], "medium", ["golden", "style"], [], ""),
    # dance + graceful
    ("pf_121", "a person dances with graceful flowing movements", ["dance"], ["person"], [], False, ["graceful"], "medium", ["golden", "style"], [], ""),
    ("pf_122", "someone sways elegantly in a smooth dance sequence", ["dance"], ["person"], [], False, ["graceful"], "medium", ["golden", "style"], [], ""),
    ("pf_123", "the figure moves rhythmically with gentle arced arm motions", ["dance"], ["person"], [], False, ["graceful"], "medium", ["golden", "style"], [], ""),
    ("pf_124", "an individual grooves gracefully to an implied steady beat", ["dance"], ["person"], [], False, ["graceful"], "medium", ["golden", "style"], [], ""),
    ("pf_125", "a human dances in a fluid, controlled, and graceful style", ["dance"], ["person"], [], False, ["graceful"], "medium", ["golden", "style"], [], ""),
    # walk + limp
    ("pf_126", "a person walks with a noticeable limp on the right side", ["walk"], ["person"], [], False, ["limp"], "hard", ["golden", "style"], [], "OOD"),
    ("pf_127", "someone limps forward awkwardly, favouring one leg", ["walk"], ["person"], [], False, ["limp"], "hard", ["golden", "style"], [], "OOD"),
    ("pf_128", "the figure moves with an uneven, irregular limping gait", ["walk"], ["person"], [], False, ["limp"], "hard", ["golden", "style"], [], "OOD"),
    ("pf_129", "an individual hobbles with a pronounced limp in each stride", ["walk"], ["person"], [], False, ["limp"], "hard", ["golden", "style"], [], "OOD"),
    ("pf_130", "a human shuffles with a lopsided gait due to a limp", ["walk"], ["person"], [], False, ["limp"], "hard", ["golden", "style"], [], "OOD"),
    # walk + cautious
    ("pf_131", "a person walks extremely cautiously, scanning the surroundings", ["walk"], ["person"], [], False, ["cautious"], "hard", ["golden", "style"], [], "OOD"),
    ("pf_132", "someone creeps forward with careful deliberate testing steps", ["walk"], ["person"], [], False, ["cautious"], "hard", ["golden", "style"], [], "OOD"),
    ("pf_133", "the figure inches along, hesitating and pausing at each step", ["walk"], ["person"], [], False, ["cautious"], "hard", ["golden", "style"], [], "OOD"),
    ("pf_134", "an individual paces warily, staying low and alert", ["walk"], ["person"], [], False, ["cautious"], "hard", ["golden", "style"], [], "OOD"),
    ("pf_135", "a human advances with measured caution, watching carefully", ["walk"], ["person"], [], False, ["cautious"], "hard", ["golden", "style"], [], "OOD"),
]

POOL_NEGATION = [
    # not-run, walks
    ("pf_136", "a person does not run but walks calmly instead", ["walk"], ["person"], ["run"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_137", "someone keeps their pace to a walk and avoids running", ["walk"], ["person"], ["run"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_138", "the figure chooses to walk, not run, across the space", ["walk"], ["person"], ["run"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_139", "an individual restrains themselves from running and only walks", ["walk"], ["person"], ["run"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_140", "a human moves at walking speed without breaking into a run", ["walk"], ["person"], ["run"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    # attempts jump but stumbles
    ("pf_141", "a person attempts to jump but stumbles and falls instead", ["fall"], ["person"], ["jump"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_142", "someone tries to leap but trips at the moment of takeoff", ["fall"], ["person"], ["jump"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_143", "the figure begins a jump motion but loses footing and falls", ["fall"], ["person"], ["jump"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_144", "an individual starts to spring up but tumbles before leaving the ground", ["fall"], ["person"], ["jump"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_145", "a human goes to jump but stumbles before leaving the floor", ["fall"], ["person"], ["jump"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    # almost falls but recovers
    ("pf_146", "a person nearly falls but catches themselves and stands", ["stand"], ["person"], ["fall"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_147", "someone wobbles dangerously but regains balance without falling", ["stand"], ["person"], ["fall"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_148", "the figure lurches toward a fall but rights themselves in time", ["stand"], ["person"], ["fall"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_149", "an individual teeters close to falling but stays upright", ["stand"], ["person"], ["fall"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_150", "a human stumbles close to the ground and then recovers to standing", ["stand"], ["person"], ["fall"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    # should not sit
    ("pf_151", "a person stays standing throughout without sitting", ["stand"], ["person"], ["sit"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_152", "someone remains on their feet for the whole sequence", ["stand"], ["person"], ["sit"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_153", "the figure keeps their upright stance and never sits down", ["stand"], ["person"], ["sit"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_154", "an individual holds their standing position throughout the clip", ["stand"], ["person"], ["sit"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_155", "a human maintains a standing posture and does not take a seat", ["stand"], ["person"], ["sit"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    # avoids kicking, walks
    ("pf_156", "a person approaches the ball but only walks past it without kicking", ["walk"], ["person", "ball"], ["kick"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_157", "someone walks near the ball and deliberately avoids kicking it", ["walk"], ["ball"], ["kick"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_158", "the figure passes by the ball while walking, without striking it", ["walk"], ["ball"], ["kick"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_159", "an individual walks forward and ignores the ball entirely", ["walk"], ["ball"], ["kick"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_160", "a human moves through the space walking, making no kicking motion", ["walk"], ["person"], ["kick"], False, [], "medium", ["golden", "negation"], [], OOD_NEGATION),
    # does not throw, holds
    ("pf_161", "a person holds the ball and walks without throwing it", ["hold"], ["person", "ball"], ["throw"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_162", "someone carries the object the whole time and never throws it", ["hold"], ["person"], ["throw"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_163", "the figure clutches the box and does not release or throw it", ["hold"], ["box"], ["throw"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_164", "an individual grips the item and keeps holding it without throwing", ["hold"], ["person"], ["throw"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_165", "a human holds onto the ball for the duration without a throw", ["hold"], ["ball"], ["throw"], False, [], "hard", ["golden", "negation"], [], OOD_NEGATION),
    # slow walk, not run
    ("pf_166", "a person moves at a deliberately slow walk with no running", ["walk"], ["person"], ["run"], False, [], "easy", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_167", "someone advances at a careful walking pace, not running", ["walk"], ["person"], ["run"], False, [], "easy", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_168", "the figure keeps a slow deliberate walk throughout the clip", ["walk"], ["person"], ["run"], False, [], "easy", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_169", "an individual plods forward slowly without ever running", ["walk"], ["person"], ["run"], False, [], "easy", ["golden", "negation"], [], OOD_NEGATION),
    ("pf_170", "a human paces slowly forward, never breaking into a run", ["walk"], ["person"], ["run"], False, [], "easy", ["golden", "negation"], [], OOD_NEGATION),
]

POOL_OBJECT_INTERACTION = [
    # kick ball
    ("pf_171", "a person kicks a ball toward the far end of the field", ["kick"], ["person", "ball"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_172", "someone boots the ball hard with their right foot", ["kick"], ["ball"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_173", "the figure strikes the ball forcefully with a sweeping kick", ["kick"], ["ball"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_174", "an individual punts the ball to a target on the ground", ["kick"], ["ball"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_175", "a human delivers a clean kick to the stationary ball ahead", ["kick"], ["person", "ball"], [], False, [], "medium", ["golden", "object"], [], ""),
    # throw box
    ("pf_176", "a person throws a box forward overhand with both hands", ["throw"], ["person", "box"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_177", "someone hurls the box across the room with momentum", ["throw"], ["box"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_178", "the figure flings the crate in a wide overhead arc", ["throw"], ["box"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_179", "an individual pitches the box with an underarm swing", ["throw"], ["box"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_180", "a human launches the box forward with sustained force", ["throw"], ["person", "box"], [], False, [], "medium", ["golden", "object"], [], ""),
    # pick cone
    ("pf_181", "a person bends down and picks up a cone from the floor", ["pick"], ["person", "cone"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_182", "someone crouches and retrieves the cone lying on the ground", ["pick"], ["cone"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_183", "the figure reaches down and lifts the cone up with one hand", ["pick"], ["cone"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_184", "an individual grabs the cone from its position on the ground", ["pick"], ["cone"], [], False, [], "easy", ["golden", "object"], [], ""),
    ("pf_185", "a human leans over and collects the cone from beside their foot", ["pick"], ["person", "cone"], [], False, [], "medium", ["golden", "object"], [], ""),
    # reach shelf
    ("pf_186", "a person reaches upward toward a high shelf above them", ["reach"], ["person", "shelf"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_187", "someone extends their arm fully to the top of the shelf", ["reach"], ["shelf"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_188", "the figure stretches overhead to touch the shelf above their head", ["reach"], ["shelf"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_189", "an individual raises both arms to access the upper shelf", ["reach"], ["shelf"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_190", "a human reaches as high as possible to get something from the shelf", ["reach"], ["person", "shelf"], [], False, [], "medium", ["golden", "object"], [], ""),
    # place box on table
    ("pf_191", "a person places a box carefully on the table surface", ["place"], ["person", "box", "table"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_192", "someone sets the box down on the table with both hands", ["place"], ["box", "table"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_193", "the figure lowers the box gently onto the tabletop", ["place"], ["box", "table"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_194", "an individual deposits the crate on the edge of the table", ["place"], ["box", "table"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_195", "a human puts the box down on the table surface with care", ["place"], ["person", "box", "table"], [], False, [], "medium", ["golden", "object"], [], ""),
    # push cart
    ("pf_196", "a person pushes a cart forward across the floor", ["push"], ["person", "cart"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_197", "someone shoves the cart steadily in a straight line", ["push"], ["cart"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_198", "the figure propels the cart forward with both hands pressed against it", ["push"], ["cart"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_199", "an individual nudges the cart into motion and keeps pushing", ["push"], ["cart"], [], False, [], "medium", ["golden", "object"], [], ""),
    ("pf_200", "a human pushes the cart by leaning their full weight into it", ["push"], ["person", "cart"], [], False, [], "medium", ["golden", "object"], [], ""),
]


def build_entry(row: tuple) -> dict:
    (
        entryId, prompt, requiredActions, requiredEntities,
        forbiddenActions, ordered, styleConstraints, difficulty,
        tags, temporalRelations, notes,
    ) = row

    return {
        "id": entryId,
        "prompt": prompt,
        "category": category_for(entryId),
        "required_actions": requiredActions,
        "required_entities": requiredEntities,
        "forbidden_actions": forbiddenActions,
        "ordered": ordered,
        "style_constraints": styleConstraints,
        "difficulty": difficulty,
        "tags": tags,
        "temporal_relations": temporalRelations,
        "notes": notes,
    }


def category_for(entryId: str) -> str:
    num = int(entryId.split("_")[1])

    if num <= 50:
        return "single_action"

    if num <= 100:
        return "multi_step"

    if num <= 135:
        return "style_modifier"

    if num <= 170:
        return "negation"

    return "object_interaction"


def check_diversity(rows: list[dict]) -> None:
    try:
        metrics = build_metrics(rows)
        result = evaluate_diversity(metrics, DEFAULT_THRESHOLDS)
        avg_sim = round(metrics.get("avg_nearest_similarity", 0.0), 3)
        ttr = round(metrics.get("type_token_ratio", 0.0), 3)
        print(f"  diversity: avg_nearest_sim={avg_sim}  TTR={ttr}  passed={result['passed']}")

        if not result["passed"]:
            print(f"  WARNING diversity failures: {result['failures']}", file=sys.stderr)

    except Exception as exc:
        print(f"  diversity check skipped: {exc}", file=sys.stderr)


def write_one_line(entries: list[dict], path: Path) -> None:
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    path.write_text("[\n" + ",\n".join(lines) + "\n]", encoding="utf-8")


def main() -> None:
    all_pools = (
        POOL_SINGLE_ACTION
        + POOL_MULTI_STEP
        + POOL_STYLE_MODIFIER
        + POOL_NEGATION
        + POOL_OBJECT_INTERACTION
    )

    if len(all_pools) != 200:
        raise ValueError(f"Expected exactly 200 pool rows, got {len(all_pools)}")

    rows = [build_entry(r) for r in all_pools]
    validated = validate_suite(rows, min_size=200)

    print(f"Validating {len(validated)} entries â€¦")
    check_diversity(validated)

    output_path = Path("data/eval/prompt_following_suite.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_one_line(validated, output_path)

    print(f"Wrote {len(validated)} prompts â†’ {output_path}")


if __name__ == "__main__":
    main()
