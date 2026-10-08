from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m scripts.inspect_aisslab_xml "
            "/path/to/Foramina_Detection"
        )

    root_dir = Path(sys.argv[1])
    xml_files = sorted(root_dir.rglob("*.xml"))

    print(f"Found {len(xml_files)} XML files")
    if not xml_files:
        raise SystemExit("No XML files found. Check the directory path.")

    for path in xml_files[:3]:
        print(f"\n=== {path} ===")

        root = ET.parse(path).getroot()
        print("Root tag:", root.tag)
        print("Filename:", root.findtext("filename"))

        for obj in root.findall("object")[:5]:
            print("\nObject fields:")
            for child in obj:
                if child.tag == "bndbox":
                    coordinates = {
                        item.tag: item.text
                        for item in child
                    }
                    print("  bndbox:", coordinates)
                else:
                    print(f"  {child.tag}: {child.text}")


if __name__ == "__main__":
    main()