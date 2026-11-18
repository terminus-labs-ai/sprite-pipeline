"""FR 7 (amendment 10): the guide line ending every direction edit's prompt,
per pronoun, stated once for the tests as the spec's literal text. It
imports nothing from production code, so a changed `stills.GUIDE_LINE`
fails the prompt tests."""

GUIDE_LINE = {
    "he": ("Image 2 is a plain grey mannequin used only as a guide: draw him in "
           "exactly the mannequin's pose, facing direction, size and position in the "
           "frame, with his feet exactly where the mannequin's feet are and his arms "
           "held out like the mannequin's. Do not draw the mannequin itself."),
    "she": ("Image 2 is a plain grey mannequin used only as a guide: draw her in "
            "exactly the mannequin's pose, facing direction, size and position in the "
            "frame, with her feet exactly where the mannequin's feet are and her arms "
            "held out like the mannequin's. Do not draw the mannequin itself."),
    "they": ("Image 2 is a plain grey mannequin used only as a guide: draw them in "
             "exactly the mannequin's pose, facing direction, size and position in "
             "the frame, with their feet exactly where the mannequin's feet are and "
             "their arms held out like the mannequin's. Do not draw the mannequin "
             "itself."),
}
