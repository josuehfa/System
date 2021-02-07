from matplotlib import pyplot as plt 
import numpy as np 
from matplotlib.animation import FuncAnimation
from skimage.draw import ellipse 
from skimage.draw import ellipse_perimeter
from skimage.draw import disk

class MapGen():
    def __init__(self, nrows, ncols, time):
        self.nrows = nrows
        self.ncols = ncols
        self.time = time
        
        self.x = []
        self.y = []
        self.z = []
        self.z_time = []
        self.obs_time = []
        self.verti_perimeters = []

    def create(self):

        nrows = self.nrows
        ncols = self.ncols
        delta_d = 1/nrows
        x = np.arange(ncols+1)*delta_d
        y = np.arange(nrows+1)*delta_d
        
        for t in range(self.time):
            z = np.zeros((nrows+1, ncols+1), dtype=np.uint8) + 1
            xx,yy = disk((0.5*nrows,0.5*nrows),0.5*nrows)
            z[xx,yy] = 10

            xx,yy = disk((0.5*nrows,0.5*nrows),0.25*nrows)
            z[xx,yy] = 20

            xx,yy = disk((0.5*nrows,0.5*nrows),0.125*nrows)
            z[xx,yy] = 50

            #Region Clearece
            xx, yy = ellipse(0.5*nrows+t*(0.5*nrows/self.time), 0.5*ncols-t*(0.5*ncols/self.time), 0.15*nrows, 0.25*ncols, rotation=np.deg2rad(10))
            x_del = np.argwhere( (xx <= 0) | (xx >= nrows) )
            y_del = np.argwhere( (yy <= 0) | (yy >= ncols) )
            xx = np.delete(xx, np.concatenate((x_del, y_del), axis=0))
            yy = np.delete(yy, np.concatenate((x_del, y_del), axis=0))
            z[xx,yy] = 50

             #Region Clearece
            xx, yy = ellipse(0.5*nrows-t*(0.5*nrows/self.time), 0.5*ncols+t*(0.5*ncols/self.time), 0.15*nrows, 0.25*ncols, rotation=np.deg2rad(10))
            x_del = np.argwhere( (xx <= 0) | (xx >= nrows) )
            y_del = np.argwhere( (yy <= 0) | (yy >= ncols) )
            xx = np.delete(xx, np.concatenate((x_del, y_del), axis=0))
            yy = np.delete(yy, np.concatenate((x_del, y_del), axis=0))
            z[xx,yy] = 50

            z = np.asarray(z,dtype=np.double) 
            self.z_time.append(z)

            
            aux_obs = []
            #z = np.zeros((nrows+1, ncols+1), dtype=np.uint8) + 1
            #Region Obstacle
            xx, yy = ellipse_perimeter(int((0.5*nrows+t*(0.5*nrows/self.time))*nrows), int((0.5*ncols-t*(0.5*ncols/self.time))*nrows), int((0.15*nrows)*nrows), int((0.25*ncols)*nrows), orientation=np.deg2rad(-10))
            x_del = np.argwhere( (xx <= 0) | (xx >= nrows*nrows) )
            y_del = np.argwhere( (yy <= 0) | (yy >= ncols*ncols) )
            xx = np.delete(xx, np.concatenate((x_del, y_del), axis=0))
            yy = np.delete(yy, np.concatenate((x_del, y_del), axis=0))
            xx = xx/nrows
            yy = yy/nrows
            xx = np.round(xx,0)
            yy = np.round(yy,0)
            xx = np.asarray(xx,dtype=np.integer) 
            yy = np.asarray(yy,dtype=np.integer)
            #z[xx,yy] = 10
            polygon = []
            for idx in range(len(xx)):
                polygon.append((xx[idx]/nrows,yy[idx]/ncols)) 
            aux_obs.append((polygon,0,0,'CB'))

            #Region Obstacle
            xx, yy = ellipse_perimeter(int((0.5*nrows-t*(0.5*nrows/self.time))*nrows), int((0.5*ncols+t*(0.5*ncols/self.time))*nrows), int((0.15*nrows)*nrows), int((0.25*ncols)*nrows), orientation=np.deg2rad(-10))
            x_del = np.argwhere( (xx <= 0) | (xx >= nrows*nrows) )
            y_del = np.argwhere( (yy <= 0) | (yy >= ncols*ncols) )
            xx = np.delete(xx, np.concatenate((x_del, y_del), axis=0))
            yy = np.delete(yy, np.concatenate((x_del, y_del), axis=0))
            xx = xx/nrows
            yy = yy/nrows
            xx = np.round(xx,0)
            yy = np.round(yy,0)
            xx = np.asarray(xx,dtype=np.integer) 
            yy = np.asarray(yy,dtype=np.integer)
            #z[xx,yy] = 10
            polygon = []
            for idx in range(len(xx)):
                polygon.append((xx[idx]/nrows,yy[idx]/ncols)) 
            aux_obs.append((polygon,0,0,'CB'))
            
            self.obs_time.append(aux_obs)
            

        #print(z)
        self.z = self.z_time[0]
        self.y = y
        self.x = x
    
    def createScenarioOne(self):
        '''
        Map of Scenario Two:
            - Create a nrow x ncol matrix with the same value for all positions.

        '''
        delta_d = 1/self.ncols
        x = np.arange(self.ncols+1)*delta_d
        y = np.arange(self.nrows+1)*delta_d
        
        for t in range(self.time):
            z = np.zeros((self.nrows+1, self.ncols+1), dtype=np.uint8) + 1
            z = np.asarray(z,dtype=np.double) 
            self.z_time.append(z)

        self.z = self.z_time[0]
        self.y = y
        self.x = x

    def createScenarioTwo(self):
        '''
        Map of Scenario Two:
            - Use a image to create a nrow x ncol matrix with values of populational density of Belo Horizonte.

        '''
        from skimage import io
        from skimage.util import invert
        from skimage.transform import rotate, resize

        #0.901 - Menor densidade
        #0.773
        #0.664
        #0.556
        #0.465 - Maior densidade

        nrows = self.nrows
        ncols = self.ncols
        delta_d = 1/nrows
        x = np.arange(ncols+1)*delta_d
        y = np.arange(nrows+1)*delta_d
    
        original = io.imread('/home/josuehfa/System/CoreSystem/ImageLib/popCalculated.png')
        image_resized = resize(original, (nrows+1,ncols+1),anti_aliasing=True)
        image_rotate = rotate(image_resized,180)
        final_image = image_rotate[:,::-1]
        final_image = np.multiply(final_image, np.where(final_image >= 0.12, 200, 1))
        final_image = final_image + 0.1        

        for t in range(self.time):
            self.z_time.append(final_image)

        self.obs_time = []
        self.z = self.z_time[0]
        self.y = y
        self.x = x

    def createScenarioThree(self, vertiports, radius):
        '''
        Map of Scenario Three:
            - Use a image to create a nrow x ncol matrix with values of populational density of Belo Horizonte.
            - Create points for safe landing and determine a radius from those points where the aircraft could operate.  

        '''
        from skimage import io
        from skimage.util import invert
        from skimage.transform import rotate, resize
        from skimage.draw import disk
        from skimage.draw import circle_perimeter
    
        nrows = self.nrows
        ncols = self.ncols
        delta_d = 1/nrows
        x = np.arange(ncols+1)*delta_d
        y = np.arange(nrows+1)*delta_d
    
        original = io.imread('/home/josuehfa/System/CoreSystem/ImageLib/popCalculated.png')
        image_resized = resize(original, (nrows+1,ncols+1),anti_aliasing=True)
        image_rotate = rotate(image_resized,180)
        final_image = image_rotate[:,::-1]
        final_image = np.multiply(final_image, np.where(final_image >= 0.1, 110, 1))

        self.vertiports = vertiports
        vertiports_map = np.zeros((nrows+1, ncols+1), dtype=np.uint8)
        for vertiport in vertiports:
            #Obtem os dados de perimetros para cada vertice
            xp,yp = circle_perimeter(int(vertiport[1]*nrows),int(vertiport[0]*nrows),int(radius*nrows))
            x_del = np.argwhere( (xp <= 0) | (xp >= nrows) )
            y_del = np.argwhere( (yp <= 0) | (yp >= ncols) )
            xp = np.delete(xp, np.concatenate((x_del, y_del), axis=0))/nrows
            yp = np.delete(yp, np.concatenate((x_del, y_del), axis=0))/nrows
            self.verti_perimeters.append([xp,yp])
            #Obtem a região dentro do circulo de perimetro
            xx,yy = disk((vertiport[1]*nrows,vertiport[0]*nrows),radius*nrows)
            x_del = np.argwhere( (xx <= 0) | (xx >= nrows) )
            y_del = np.argwhere( (yy <= 0) | (yy >= ncols) )
            xx = np.delete(xx, np.concatenate((x_del, y_del), axis=0))
            yy = np.delete(yy, np.concatenate((x_del, y_del), axis=0))
            vertiports_map[xx,yy] = 1
         
        mask = vertiports_map < 1
        mapimage = final_image*vertiports_map + mask*100
        mapimage = mapimage + 0.1

        for t in range(self.time):
            self.z_time.append(mapimage)

        self.obs_time = []
        self.z = self.z_time[0]
        self.y = y
        self.x = x

    def createScenarioFour(self):
        '''
        Map of Scenario Four:
            - Use a meteorological data as cost
        '''
        import os
        from skimage import io
        from skimage.util import crop
        from skimage.util import invert
        from skimage.color import rgb2gray
        from skimage.transform import rotate, resize
    
        nrows = self.nrows
        ncols = self.ncols
        delta_d = 1/nrows
        x = np.arange(ncols+1)*delta_d
        y = np.arange(nrows+1)*delta_d

        # Dados do radar meteorologico de Gama/DF (5 elemento de retorno) obtidos na redemet
        # (result = redemet.get_produto_radar(tipo='maxcappi',area='tm',data='2021011414',anima=5))
        mypath = '/home/josuehfa/System/CoreSystem/MeteorologicalData/'
        (_, _, filenames) = next(os.walk(mypath))
        filenames = sorted(filenames)
        time = len(filenames)

        z_time = []
        for t in range(self.time):

            original = io.imread(os.path.join(mypath, filenames[t]))
            image_resized = rgb2gray(resize(original, (2*nrows+1,2*ncols+1),anti_aliasing=True))
            image_croped = crop(image_resized, ((nrows/2, nrows/2), (nrows/2, nrows/2)), copy=False)
            image_rotate = rotate(image_croped,180)
            final_image = image_rotate[:,::-1]
            final_image = np.multiply(final_image, np.where(final_image >= 0.1, 110, 1))
            self.z_time.append(final_image)
        

        self.obs_time = []
        self.z = self.z_time[0]
        self.y = y
        self.x = x

    def createScenarioFive(self):
        '''
        Map of Scenario Five:
            - Use a image to create a nrow x ncol matrix with values of populational density of Belo Horizonte.
            - Use a meteorological data as cost
        '''
        import os
        from skimage import io
        from skimage.util import crop
        from skimage.util import invert
        from skimage.color import rgb2gray
        from skimage.transform import rotate, resize
    
        nrows = self.nrows
        ncols = self.ncols
        delta_d = 1/nrows
        x = np.arange(ncols+1)*delta_d
        y = np.arange(nrows+1)*delta_d


        # Dados do radar meteorologico de Gama/DF (5 elemento de retorno) obtidos na redemet
        # (result = redemet.get_produto_radar(tipo='maxcappi',area='tm',data='2021011414',anima=5))
        mypath = '/home/josuehfa/System/CoreSystem/MeteorologicalData/'
        (_, _, filenames) = next(os.walk(mypath))
        filenames = sorted(filenames)
        time = len(filenames)

        z_time = []
        for t in range(self.time):

            original_met = io.imread(os.path.join(mypath, filenames[t]))
            image_resized_met = rgb2gray(resize(original_met, (2*nrows+1,2*ncols+1),anti_aliasing=True))
            image_croped_met = crop(image_resized_met, ((nrows/2, nrows/2), (nrows/2, nrows/2)), copy=False)
            image_rotate_met = rotate(image_croped_met,180)
            final_image_met = image_rotate_met[:,::-1]
            final_image_met = np.multiply(final_image_met, np.where(final_image_met >= 0.1, 110, 1))

            original_pop = io.imread('/home/josuehfa/System/CoreSystem/ImageLib/popCalculated.png')
            image_resized_pop = resize(original_pop, (nrows+1,ncols+1),anti_aliasing=True)
            image_rotate_pop = rotate(image_resized_pop,180)
            final_image_pop = image_rotate_pop[:,::-1]
            final_image_pop = np.multiply(final_image_pop, np.where(final_image_pop >= 0.1, 110, 1))

            final_image = 1.4*final_image_met + final_image_pop
            final_image = final_image + 0.1       



            self.z_time.append(final_image)
        

        self.obs_time = []
        self.z = self.z_time[0]
        self.y = y
        self.x = x




    def createFromMap(self):
        from skimage.transform import rotate
        from skimage.transform import resize
        from skimage.color import rgb2hsv
        from skimage.util import invert
        from skimage import io

        nrows = self.nrows
        ncols = self.ncols
        delta_d = 1/nrows
        x = np.arange(ncols+1)*delta_d
        y = np.arange(nrows+1)*delta_d

        #original = data.astronaut()
        original = io.imread('/home/josuehfa/System/CoreSystem/ImageLib/TestMap2.png')[:,:,:3]
        image_resized = resize(original, (nrows+1,ncols+1),anti_aliasing=True)
        
        hsv_img = rgb2hsv(image_resized)
        sat_img = hsv_img[:, :, 1]*100
        image_rotate = rotate(sat_img,180)
        final_image = image_rotate[:,::-1]
        final_image = invert(final_image, True)+100
        segment = final_image > 86
        final_image = final_image * (segment+0.01)

        for t in range(self.time):
            self.z_time.append(final_image)

        self.obs_time = []
        self.z = self.z_time[0]
        self.y = y
        self.x = x
        
        #import matplotlib.pyplot as plt
        #fig,ax0 = plt.subplots(ncols=1, figsize=(8, 2))
        #ax0.imshow(final_image)
        #ax0.set_title("RGB image")
        #ax0.axis('off')
        #fig.tight_layout()
        #plt.show()

    def plot_map(self, t, axis):
        #Obstacle
        aux_im = []
        aux_im.append(axis.pcolormesh(self.x, self.y, self.z_time[t]*0.01, cmap='rainbow', shading='nearest'))
        if self.obs_time != [] :
            for idx, polygon in enumerate(self.obs_time[t]):
                lat,lon = zip(*polygon[0])
                lat = list(lat)
                lon = list(lon)
                lat = np.asarray(lat,dtype=np.double)
                lon = np.asarray(lon,dtype=np.double)
                #aux_im.append(axis.plot(lon, lat, linestyle='-', color='red'))
                #aux_im.append(axis.fill(lon, lat, facecolor='gray', edgecolor='black'))
                #aux_im[idx+1] = aux_im[idx+1][0]
            
        return aux_im

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import pyplot
    ims = []
    time = 10
    nrows = 150
    ncols = 150
    delta_d = 1/nrows
    fig = plt.figure()
    axis = plt.axes(xlim =(0, 1),  
                    ylim =(0, 1))
    
    vertiports = [(0.123, 0.816), (0.19, 0.639), (0.289, 0.487), (0.355, 0.294), (0.543, 0.228), (0.737, 0.195), (0.9, 0.187)]
    radius = 0.2
    mapgen = MapGen(nrows, ncols,time)
    #mapgen.create()
    #mapgen.createScenarioThree(vertiports,radius)
    mapgen.createScenarioTwo()
    #mapgen.createEmptyMap()
    #mapgen.createFromMap()
    #mapgen.plot_map(axis)
    #mapgen.createScenarioFive()
    
    
    for t in range(time):
        #Obstacle
        aux_im = []
        aux_im.append(axis.pcolormesh(mapgen.x, mapgen.y, mapgen.z_time[t]*delta_d, cmap='rainbow', shading='nearest'))
        if mapgen.obs_time != [] :
            for idx, polygon in enumerate(mapgen.obs_time[t]):
                lat,lon = zip(*polygon[0])
                lat = list(lat)
                lon = list(lon)
                #lat.append(polygon[0][0][0])
                #lon.append(polygon[0][0][1])
                lat = np.asarray(lat,dtype=np.double)
                lon = np.asarray(lon,dtype=np.double)
                aux_im.append(axis.plot(lon, lat, linestyle='-', color='red'))
                #aux_im.append(axis.fill(lon, lat, facecolor='gray', edgecolor='black'))
                aux_im[idx+1] = aux_im[idx+1][0]
        
        test = tuple(aux_im)
        ims.append(test)
        #aux_im.append(axis.pcolormesh(mapgen.x, mapgen.y, mapgen.z_time[t]*delta_d, cmap='RdBu', shading='nearest', vmin=-5, vmax=5))
        #ims.append(tuple((aux_im,)))
        #ims.append((axis.pcolormesh(mapgen.x, mapgen.y, mapgen.z_time[t]*delta_d, cmap='RdBu', shading='nearest', vmin=-5, vmax=5),))
        pyplot.clf()

    
    #axis.pcolormesh(x, y, z*0.02, cmap='RdBu', shading='nearest', vmin=-5, vmax=5)
    #line, = axis.plot([], [],'.', lw = 3) 

    # data which the line will  
    # contain (x, y) 
    #def init():  
    #    line.set_data([], []) 
    #    return line, 

    #x = np.linspace(0, 4, 1000)
    #y = np.sin(2 * np.pi * (x - 0.01))
    #def animate(i): 
    
        # plots a sine graph 
         
    #    line.set_data(path_y[:i], path_x[:i]) 
        
    #    return line, 
    
    im_ani = animation.ArtistAnimation(fig, ims, interval=10000/time, blit=True)
    #anim = FuncAnimation(fig, animate, init_func = init, 
    #                    frames =len(path_y) , interval = 200, blit = True) 
    
    plt.show()
    