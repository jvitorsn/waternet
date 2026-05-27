import cairosvg, re

with open("./waternet_pipeline.svg", "r") as f:
    svg = f.read()

# Inject white background rect right after the opening <svg ...> tag
svg = re.sub(
    r'(<svg[^>]*>)',
    r'\1<rect width="100%" height="100%" fill="white"/>',
    svg, count=1
)

cairosvg.svg2png(
    bytestring=svg.encode(),
    write_to="/waternet_pipeline.png",
    scale=3          # 3× for high-res / print quality
)
print("Done")