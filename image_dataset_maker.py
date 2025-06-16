from selenium import webdriver
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup, Tag
import time, requests, base64
from PIL import Image
from io import BytesIO
import open_clip, torch
from tqdm.auto import tqdm



def src_to_base64(src):
    if src.startswith('data:image'):
        return src

    res = requests.get(src, headers={'User-Agent': 'Mozilla/5.0'})
    return base64.b64encode(res.content).decode('utf-8')



class Client():
    def __init__(self, path, headless=True):

        self.service = Service(executable_path=path)
        self.options = Options()
        self.browser = webdriver.Firefox

        if headless:
            self.options.add_argument('--headless')

    def get_images(self, query):

        browser = self.browser(service=self.service, options=self.options)
        browser.get(f'https://www.google.com/search?q={query}&source=hp&sclient=img&udm=2')


        while True:

            browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")

            images = self._get_image_elements_from_source(browser.page_source)

            WebDriverWait(browser, 10).until(lambda d: browser.execute_script("return Array.from(document.images).every(img => img.complete);"))
            time.sleep(0.5)

            if len(self._get_image_elements_from_source(browser.page_source)) == len(images): break

        browser.close()

        base64_images = []
        for img in tqdm(images, desc='processing images...'):
            if isinstance(img, Tag):
                try:
                    base64_images.append(src_to_base64(img.get('src')))
                except: pass

        return base64_images

    def _get_image_elements_from_source(self, source):
        soup = BeautifulSoup(source, 'html.parser')
        return soup.find_all(lambda tag: tag.has_attr('class') and tag.get('class') == ['YQ4gaf'])



class Classfication():
    def __init__(self, image_ref: list[str], text_ref: list[str]):

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.reference_vector = self._create_reference_vector(image_ref, text_ref)

    def embed_image(self, base64_text):
        img = self._base64_to_image(base64_text)
        return self.model.encode_image(self.preprocess(img).unsqueeze(0))

    def embed_text(self, text):
        return self.model.encode_text(self.tokenizer([text]))


    def _base64_to_image(self, base64_text: str):
        if ',' in base64_text: base64_text = base64_text.split(',')[1]
        return Image.open(BytesIO(base64.b64decode(base64_text)))

    def _create_reference_vector(self, image_ref, text_ref):
        image_embed = [self.embed_image(src_to_base64(i)) for i in image_ref]
        text_embed  = [self.embed_text(i) for i in text_ref]

        ref = image_embed + text_embed
        ref = torch.cat(ref).mean(dim=0, keepdim=True)
        ref /= ref.norm(dim=-1, keepdim=True)
        return ref.T

    def classify(self, base64_text_test):
        img = self.embed_image(base64_text_test)
        img /= img.norm(dim=-1, keepdim=True)

        return (img @ self.reference_vector).item()

    def classify_all(self, base64_text_tests: list[str], threshold=0.5):

        result = []
        count = 0

        iter = tqdm(base64_text_tests, desc='classify images...')

        for b64_text in iter:
            if self.classify(b64_text) > threshold:

                count += 1
                result.append(b64_text)
                iter.set_postfix({'match' : count})

        return result

    def save_base64_image(self, base64_text, path):

        if ',' in base64_text: base64_text = base64_text.split(',')[1]

        img = base64.b64decode(base64_text)
        img = Image.open(BytesIO(img))
        img.save(path)



if __name__ == '__main__':

    # create Classfication Class based on ViT-32
    cf = Classfication(
        ['https://{The program automatically crawls only images that are similar to the input image.}'],
        ['frog']
    )

    # create Firefox client
    client = Client(path='geckodriver path')

    # get images from google
    images = client.get_images(query='pepe') # Query are required

    # classify all images
    result = cf.classify_all(images)

    # save images
    for idx, img in enumerate(result):
        cf.save_base64_image(img, f'./{idx}.png')
