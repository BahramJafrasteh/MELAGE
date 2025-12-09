from PIL import Image, ImageDraw

# --- Configuration ---
# You can change these colors and sizes to match your application's theme
CONFIG = {
    "size": 16,  # Standard resolution size
    "colors": {
        "border": (100, 100, 100),
        "fill_checked": (40, 120, 255),
        "fill_pressed": (200, 200, 200),
        "checkmark": (255, 255, 255),
        "indeterminate_mark": (255, 255, 255),
        "focus_glow": (70, 150, 255),
        "disabled_border": (180, 180, 180),
        "disabled_fill": (220, 220, 220),
        "disabled_mark": (140, 140, 140),
    }
}


def create_checkbox_icon(filename, size, state, sub_state):
    """Generates and saves a single checkbox icon image."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))  # Transparent background
    draw = ImageDraw.Draw(img)
    c = CONFIG["colors"]

    # Define coordinates for the box
    box = [(2, 2), (size - 3, size - 3)]

    # --- Draw Focus Glow (if applicable) ---
    if sub_state == "focus":
        draw.rectangle([(0, 0), (size - 1, size - 1)], fill=None, outline=c["focus_glow"], width=2)

    # --- Determine Colors based on Disabled State ---
    border_color = c["disabled_border"] if sub_state == "disabled" else c["border"]
    fill_color = c["fill_checked"]
    mark_color = c["checkmark"]

    if sub_state == "disabled":
        fill_color = c["disabled_fill"]
        mark_color = c["disabled_mark"]

    # --- Draw Box and Fill based on State ---
    bg_fill = c["fill_pressed"] if sub_state == "pressed" else None

    if state == "checked":
        draw.rectangle(box, fill=fill_color, outline=border_color, width=1)
    else:
        draw.rectangle(box, fill=bg_fill, outline=border_color, width=1)

    # --- Draw Mark (Checkmark or Dash) ---
    if state == "checked":
        # Checkmark points
        p1 = (size * 0.25, size * 0.5)
        p2 = (size * 0.45, size * 0.7)
        p3 = (size * 0.75, size * 0.3)
        draw.line([p1, p2, p3], fill=mark_color, width=int(size * 0.12))

    elif state == "indeterminate":
        # Indeterminate dash
        dash_y = size / 2
        draw.line([(size * 0.25, dash_y), (size * 0.75, dash_y)], fill=mark_color if state == "checked" else fill_color,
                  width=int(size * 0.12))

    img.save(filename)


if __name__ == '__main__':
    # List of all states and sub-states to generate
    states = ["unchecked", "checked", "indeterminate"]
    sub_states = ["", "_focus", "_pressed", "_disabled"]

    print("Generating checkbox images...")

    for state in states:
        for sub_state_str in sub_states:
            # Standard resolution
            filename_std = f"checkbox_{state}{sub_state_str}.png"
            create_checkbox_icon(filename_std, CONFIG["size"], state, sub_state_str.strip("_"))

            # High resolution (@2x)
            filename_2x = f"checkbox_{state}{sub_state_str}@2x.png"
            create_checkbox_icon(filename_2x, CONFIG["size"] * 2, state, sub_state_str.strip("_"))

    print("Generation complete.")