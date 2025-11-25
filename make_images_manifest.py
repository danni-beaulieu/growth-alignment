import os
import json

def create_images_json(directory, output_file="images.json"):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter only image files (you can adjust extensions as needed)
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}
    image_files = [
        f for f in files
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in image_extensions
    ]

    # Write to JSON
    with open(output_file, "w") as f:
        json.dump(image_files, f, indent=2)

    print(f"Created {output_file} with {len(image_files)} filenames.")

# Example usage:
# create_images_json("/path/to/your/images")

create_images_json("/Users/b.danni/Projects/growth-display/public/images", output_file="images.json")
