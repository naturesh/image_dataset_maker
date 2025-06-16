# Image Dataset Maker 
- **The easiest way to build an image dataset**

## *Desc*
  - We make one reference vector by normalizing the vector of the reference image, text, and compare the cosine similarity, where the vector embedding uses the ViT-32 model.
  - Use all images from query's Google Image search results.
  - Too many image search results may cause errors.

## Result 


<table>
  <tr>
    <td><h3><b>Reference Image</b></h1></td>
    <td><h3><b>After crawl & classify</b></h3></td>
  </tr>

  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/f818be5c-1fd6-46a4-a54b-3dfa08190320" width="400px" height="400px"> 
    </td>
    <td>
        <img src="https://github.com/user-attachments/assets/3f63d230-9027-4ddd-b6d0-e642f9fd4ad8" width="400px" height="400px"> 
    </td> 
  </tr>
</table>

## How to use

```python

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

```
