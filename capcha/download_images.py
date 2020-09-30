# USAGE
# python download_images.py --output downloads

# import the necessary packages
import argparse
import requests
import time
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
ap.add_argument("-n", "--num-images", type=int,
	default=500, help="# of images to download")
args = vars(ap.parse_args())

import json
def convert_string_to_dict(data):
	return  json.loads(data)

# initialize the URL that contains the captcha images that we will
# be downloading along with the total number of images downloaded
# thus far
# url = "https://www.e-zpassny.com/vector/jcaptcha.do"
url = "https://apivtp.vietteltelecom.vn:6768/myviettel.php/getCaptcha"
total = 0
r_source = requests.get(url)
json_result = r_source.content.decode('utf-8')

result_dict = convert_string_to_dict(json_result)
gen_captcha_url = result_dict['data']['url']

if not result_dict['errorCode']:
	# loop over the number of images to download
	for i in range(0, args["num_images"]):
		try:
			# try to grab a new captcha image
			r = requests.get(gen_captcha_url, timeout=10)

			# save the image to disk
			p = os.path.sep.join([args["output"], "{}.jpg".format(
				str(total).zfill(5))])
			f = open(p, "wb")
			f.write(r.content)
			f.close()

			# update the counter
			print("[INFO] downloaded: {}".format(p))
			total += 1

		# handle if any exceptions are thrown during the download process
		except:
			print("[INFO] error downloading image...")

		# insert a small sleep to be courteous to the server
		time.sleep(0.1)
