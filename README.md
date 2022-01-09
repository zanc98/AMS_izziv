# AMS_izziv

## AMS izziv: Izločitev razmejitvene ravnine, segmentacija korpus kalozuma in merjenje profila debeline

### Datoteka "ams_izziv_64170053.py" vsebuje naslednje tri funkcije:
	1. Funkcijo za izločanje korpus kalozuma v 2D T1-uteženi sredinski rezini "midsaggital_plane_extraction(t1_image)"
	Funkcija na vhodu sprejme 3D MR T1-uteženo sliko v spremenljiki t1_image tipa SimpleITK.Image in v spremenljivki t1_slice tipa SimpleITK.Image
	vrne 2D stransko sredinsko rezino, z metodo interpolacije sagitalnga prereza v centru mase.

	2. Funkcijo za razgradnjo korpus kalozuma v 2D T1-uteženi sredinski rezini "corpus_callosum_segmentation(t1_slice)"
	Funkcija na vhodu sprejme 2D MR T1-uteženo stransko sredinsko rezino v spremenljivki t1_slice tipa SimpleITK.Image in v spremenljivki cc_seg
	tipa SimpleITK.Image vrne pripadajoco razgradnjo korpus kalozuma.
	Za zagon te funkcije potrebujete dve datoteki z utežmi modelov: "model0-corpuss_callosum8_swirl5.h5" in "model1-corpuss_callosum8_swirl5.h5"

	3. Funkcijo za izločanje korpus kalozuma v 2D T1-uteženi sredinski rezini "midsaggital_plane_extraction_ellipse(t1_image)"
	Funkcija na vhodu sprejme 3D MR T1-uteženo sliko v spremenljiki t1_image tipa SimpleITK.Image in v spremenljivki t1_slice tipa SimpleITK.Image
	vrne 2D stransko sredinsko rezino z metodo prileganja elipse.

### Datoteka "CC_thickness_profile_funkcija" vsebuje funkcijo za določitev profile debeline v 2D binarni stranski rezini razgradnje korpus kalozuma:
	Funkcija CC_thickness_profile(iImage, direction) na vhodu sprejme 2D binarno sliko maske korpus kalozuma in znakovni niz 'up' ali 'down' za primer,
	ko je CC lok obrnjen navzgor ali pa navzdol. Funkcija izriše sliko maske z središčnico in označenima začetno in končno točko ter graf profila debeline
	korpus kalozumaVrne spremenljivko "debelina" tipa numpy array, ki predstavlja profil debeline korpus kalozuma


## Funkcije so napisane v progrmsekm jeziku python, za izvajanje funkcij pa so potrebni moduli:
numpy, matplotlib, SimpleITK, scipy, tqdm, random, sklearn, skimage, keras in tensorflow
