from osgeo import gdal

dem = gdal.Open(r"D:\MAS_DataScience\Luftbilder_Swisstopo_10_10\swissimage-dop10_2021_2665-1258_0.1_2056.tif")
gt = dem.GetGeoTransform()
print(gt)
print("Band count:", dem.RasterCount)

xmin = gt[0]
ymax = gt[3]
res = gt[1]

xlen = res * dem.RasterXSize
ylen = res * dem.RasterYSize

print(xlen)
print(ylen)

div = 10

xsize = xlen/div
ysize = ylen/div

xsteps = [xmin + xsize * i for i in range(div+1)]
ysteps = [ymax - ysize * i for i in range(div+1)]

print(xsteps)
print(ysteps)

for i in range(div):
    for j in range(div):
        xmin = xsteps[i]
        xmax = xsteps[i+1]
        ymax = ysteps[j]
        ymin = ysteps[j+1]

        print('xmin: ' + str(xmin))
        print('xmax: ' + str(xmax))
        print('ymin: ' + str(ymin))
        print('ymax: ' + str(ymax))
        print('\n')

        #gdal.Warp('dem' + str(i) + str(j) + '.tif', dem,
                  #outputBounds=(xmin, ymin, xmax, ymax), dstNodata=-9999)

        #gdal.Translate('D:/MAS_DataScience/Luftbilder_Swisstopo_10_10_splitted/' + 'swissimage-dop10_2021_2665-1258_0.1_2056' + str(i) + str(j) + '.tif', dem,
                       #projWin = (xmin, ymax, xmax, ymin))