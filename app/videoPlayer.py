import streamlit.components.v1 as components

html = """<!DOCTYPE html>
<html>
<head>
    <title>Video Player with Subtitles</title>
</head>
<body>

    <input type="file" id="fileInput" accept="video/*" />
    <br />
    <input type="file" id="subtitleInput" accept=".srt" />
    <br />
    <video id="videoPlayer" controls style="width: 100%; max-width: 600px;">
        Your browser does not support HTML5 video.
    </video>

    <script>
        document.getElementById('fileInput').addEventListener('change', function(e) {
            var file = e.target.files[0];
            var url = URL.createObjectURL(file);
            document.getElementById('videoPlayer').src = url;
        });

        document.getElementById('subtitleInput').addEventListener('change', function(e) {
            var file = e.target.files[0];
            var reader = new FileReader();

            reader.onload = function(e) {
                var subtitles = parseSRT(e.target.result);
                // You can now use the subtitles variable to display subtitles on the video.
                // However, implementing a full SRT parser and synchronizer is complex and beyond the scope of this example.
                console.log(subtitles);
            };

            reader.readAsText(file);
        });

        function parseSRT(data) {
            // This is a very basic parser and does not handle all SRT features.
            var regex = /(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\n*$)/g;
            var subtitles = [];
            var match;

            while (match = regex.exec(data)) {
                subtitles.push({
                    index: match[1],
                    startTime: match[2],
                    endTime: match[3],
                    text: match[4]
                });
            }

            return subtitles;
        }
    </script>

</body>
</html>
"""


def get_video_iframe():
    return components.html(html=html)
