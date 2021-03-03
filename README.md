# scanDeskew-Flask
## TODO:
- find a better way and better parameters to treat the image
    (to reduce errors of not finding enough contours later)
- !!! create a better algorithm to find contours
## Requête-type CURL:
```shell
curl -X PUT -H "Authorization: <token>" -F "image=<path_to_image>" http://localhost:5000/scanDeskew --output <image_output>
```
Exemple:
```shell
curl -X PUT -H "Authorization: 8bf54562-3000-11eb-adc1-0242ac120002" -F "image=@/home/axel/Pictures/DS_MemoraCar/greyCard-1.jpg" http://localhost:5000/scanDeskew --output scanned_image.jpg
```
## Réponses de l'API
- ### Code 200
Requête traitée avec succès
- ### Code 501
L'image n'a pas pu être traité
(Aucun contour utilisable n'a été trouvé)