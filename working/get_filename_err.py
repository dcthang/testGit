

html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <title>MathJax example</title>
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async
          src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
  </script>
  <style>
	html *
		{
		font-size: 1em !important;
		color: #000 !important;
		font-family: Arial !important;
		}
	img {
		max-width: 300px;
		height: auto;
		}
	textarea {
	max-width: 100px;
	}
	</style>
</head>
<body>
<table>
<tr>
    <td>Cong thuc</td>
    <td>Anh gen</td>
    <td>Anh goc</td>
</tr>


"""

footer = """
</table>
</body>
</html>
"""

number_file = 1
start_line = 6933
end_line = 6934
# 30000
mathpix_file = r"./error4.txt"
output_file = r"./error4_out_{0}_check.html"


mathpix_file = open(mathpix_file, "r")
i = 0

while True:
    line = mathpix_file.readline()
    if not line:
        break

    # split line to image file and latex
    i += 1
    # if i > end_line or i < start_line:
    #     continue
    _tm = line.split("\t")
    latex = _tm[2].rstrip()
    filename = _tm[1]
    html += "<tr><td><textarea>{0}</textarea></td><td> $${1}$$ </td><td><img src=\'./images4/{2}.jpg\'></td>".format(
        line, latex, filename)
    file_index = i % number_file
    o_file = open(output_file.format(file_index), "w")
    o_file.write(html+footer)
    o_file.close()
