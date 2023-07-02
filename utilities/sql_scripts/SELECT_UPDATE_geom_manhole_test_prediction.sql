SELECT 
CONCAT(
	'UPDATE public.manhole_test_prediction SET geom = st_geomfromtext(',
	'''',
	'POINT(',
	mtp.x,
	' ',
	mtp.y,
	')',
	'''',
	', 2056) WHERE image_name = ',
	'''',
	mtp.image_name, '''',
	' AND index = ',
	mtp.index, ';')
FROM public.manhole_test_prediction mtp