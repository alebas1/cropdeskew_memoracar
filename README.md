# Crop and deskew module for MemoraCar's OCR
In collaboration with MemoraCar @ https://www.memoracar.com/
## cURL request:
Just for the demo, we use the cURL command on linux.
```shell
curl -X PUT -H "Authorization: <token>" -F "image=<path_to_image>" <adress> --output <image_output>
```
Example:
```shell
curl -X PUT -H "Authorization: 8bf54562-3000-11eb-adc1-0242ac120002" -F "image=@/home/axel/Pictures/DS_MemoraCar/greyCard-1.jpg" http://localhost:5000/scanDeskew --output scanned_image.jpg
```
## Réponses de l'API
- ### Code 200
  - Image treated successfully, returning the treated image.
- ### Code 501
  - Couldn't treat the image, returning the untreated image.
## TODO:
- potentially make the code more readable than it is (it is readable but can be improved)
  - Using classes?
  - Reformatting the project structure?