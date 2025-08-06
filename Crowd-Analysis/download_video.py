import urllib.request

url = "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"
filename = "input_video.mp4"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as resp, open(filename, 'wb') as f:
    f.write(resp.read())

print("✅ Downloaded 'input_video.mp4'")
