import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

titles=['blue','green','red']
off= 400
subplot_col = 4
subplot_row = 5

# images = (45, 250, 1000) # j, i
# images = (400, 640, 1600) # p
images = (200, 800, 4000)

color_g = []
# color_g = [0.3479179141328855, 2.2564887647925436, 3.5311952630388665] # camera
# color_g = [2.0923762339482535, 2.2983849203788567, 2.307665215570257] # android
# color_g = [1.6333170532734023, 3.37945072534680, 4.072643464819345] # joshua - j
# color_g = [2.686591551386053, 3.4169576028486985, 2.758177317545983] # iphone - i
# color_g = [1.8225188425112944, 4.439699392384591, 26.16611057711184] # camera - c
# color_g = [3.6333170532734023, 3.67945072534680, 3.672643464819345] # check

a_values = []

for index, pixel in enumerate(images):
  a_values.append(images[len(images) - 1] / float(images[index]))

color_channels = ('b', 'g', 'r')
from_color_array = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

def b_g_channel_function(b, channel):
  return np.power(b, color_g[channel])

def copy_image_size(image):
  return np.zeros((image.shape[0], image.shape[1], 3), np.float32)

def is_pixel_saturated(image, height, width):
  threshold_value = 255
  return image.item(height, width, 0) >= threshold_value or image.item(height, width, 1) >= threshold_value or image.item(height, width, 2) >= threshold_value

def create_histograms(image, location, name, tree_size, bin_size, use_g):
  for color_index, color in enumerate(color_channels):
    range_max = 255
    if use_g:
      range_max = b_g_channel_function(256, color_index)
    histr = cv2.calcHist([image], [color_index], None, [bin_size], [0, range_max])

    plt.figure(1)
    plt.plot(histr, color = color)
    plt.figure(2)
    plt.plot(histr, color = color)
    plt.figure(1)

    plt.xlim([0, 256])
    # plt.ylim([0, 256])
    plt.savefig(location + color + '_' + name + '.png', bbox_inches='tight')
    plt.close()

  plt.figure(2)
  plt.xlim([0, 256])
  plt.savefig(location + 'all_' + name + '.png', bbox_inches='tight')
  plt.close()

def make_images_linear(image):
  new_image = copy_image_size(image)
  for height in range(image.shape[0]):
    for width in range(image.shape[1]):
      if not is_pixel_saturated(image, height, width):
        for channel in range(len(color_channels)):
          b = image.item(height, width, channel)
          new_image.itemset((height, width, channel), b_g_channel_function(b, channel))
  return new_image

def get_color_calibration_images(time):
  calibration_images = []
  # Read the image files 'p' for phone and 'w' for Nikon pictures
  for n in range(1,len(time) + 1):
    calibration_images.append(cv2.imread('./calibrationPhotos/p/p'+str(n)+'.JPG'))

  plot_original_images(calibration_images, time)

  blue=[]
  green=[]
  red=[]

  for img in calibration_images:
      b, g, r = cv2.split(img)
      blue.append(b)
      green.append(g)
      red.append(r)

  return [blue,green,red]

def plot_original_images(original_img, time):
  plt.figure()
  for n, img in enumerate(tuple(original_img)):
      plt.subplot(subplot_col,subplot_row,n+1), plt.imshow(img,'gray')
      plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
      plt.suptitle("Radiometric Calibration Images, G=200"")
      plt.title("T="+ str(time[n]))
      plt.savefig('./results/part_one/original_img.png', bbox_inches='tight')
  plt.close()

def plot_channel_calibration_histograms(color_channel_images, color_channel_index):
  plt.figure()
  for n, img in enumerate(tuple(color_channel_images)):
      plt.subplot(subplot_col,subplot_row,n+1),
      plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
      plt.hist(img.ravel(),256,[0,256])
      plt.suptitle("Histogram of Images")
      plt.title("Col="+ str(titles[color_channel_index]) + " img=%d" %n)
      plt.savefig('./results/part_one/histogram_col-'+ str(titles[color_channel_index])+'.png', bbox_inches='tight')
  plt.close()

def plot_masked_images(masked_img, color_channel_index):
  plt.figure()
  for n, img in enumerate(tuple(masked_img)):
      plt.subplot(subplot_col,subplot_row,n+1), plt.imshow(img,'gray')
      plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
      plt.suptitle("Masked Img")
      plt.title("Col= "+ str(titles[color_channel_index])+" img=%d" %n)
      plt.savefig('./results/part_one/masked_img-'+ str(titles[color_channel_index])+'.png', bbox_inches='tight')
  plt.close()

def plot_masked_images_histogram(hist_mask, color_channel_index):
  plt.figure()
  for n, img in enumerate(tuple(hist_mask)):
    plt.subplot(subplot_col,subplot_row,n+1), plt.plot(img)
    plt.suptitle("Center Pixels of Images")
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
    plt.xlim([0,256])
    plt.title('Col=' + str(titles[color_channel_index])+ " img=%d" %n)
    plt.savefig('./results/part_one/mask_hist-'+ str(titles[color_channel_index])+'.png', bbox_inches='tight')
  plt.close()

def save_cropped_images(crop_img, color_channel_index):
  plt.figure()
  for n, img in enumerate(tuple(crop_img)):
    plt.subplot(subplot_col,subplot_row,n+1), plt.imshow(img,'gray')
    plt.suptitle("Cropped Images")
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
    plt.title('Col=' + str(titles[color_channel_index])+" img=%d"%n)
    plt.savefig('./results/part_one/crop_img-'+ str(titles[color_channel_index])+'.png', bbox_inches='tight')
  plt.close()

def estimate_g_by_b(mu_img, time, color_channel_index):
  plt.figure()
  plt.plot(time,mu_img,'b',time,mu_img,'*'),
  plt.title("Estimation for parameter g from B'(T)' Col="+ str(titles[color_channel_index]))
  plt.ylabel("Pixel Intesity B(T)")
  plt.xlabel("T(s)")
  plt.legend(loc='upper left')
  plt.savefig('./results/part_one/g_estimate-'+ str(titles[color_channel_index])+'.png', bbox_inches='tight')
  plt.close()

def plot_log_b_for_t(mu_img, time, color_channel_index):
  plt.figure()
  logB=np.log10(mu_img)
  logT=np.log10(time)
  slope2, intercept2, r2,p2,sigma2=linregress(logT,logB)
  a=slope2
  g=1/a
  b=intercept2
  ylog=b + a*logT
  plt.plot(logT,logB,'b',logT,ylog,'r', logT,logB,'*', label ='y=%.2fx + %.2f'%(a,b))
  plt.title("Estimation for parameter g from Log(B'(T)') Col="+ str(titles[color_channel_index]))
  plt.ylabel("Pixel Intesity Log(B(s)')")
  plt.xlabel("Log(T(s))")
  plt.legend(loc='upper left')
  plt.savefig('./results/part_one/g_estimate_log-'+ str(titles[color_channel_index]) +'.png', bbox_inches='tight')
  
  print("g=", g)
  with open("RGB_estimation_results.txt", "a") as file:
      file.write(str(titles[color_channel_index])+ "\n" + "g=%f a=%f b=%f\n"%(g,slope2,intercept2))
  plt.close()
  return g

def b_a_estimation(mu_img, g, time, color_channel_index):
  plt.figure()
  Bg=np.power(mu_img,g)
  plt.plot(time,Bg)
  plt.title("Estimation for parameter g from B'=B^1/a Col=" + str(titles[color_channel_index]))
  plt.ylabel("Pixel Intesity B=B^1/a")
  plt.xlabel("T(s)")
  plt.savefig('./results/part_one/g_estimate_a-'+ str(titles[color_channel_index])+'.png', bbox_inches='tight')
  plt.close()

def part_one():
  #File to output the values of a and g based on the estimation
  with open("./RGB_estimation_results.txt", "w") as file:
      file.write("#Estimation of parameter g from B'(T)\n#CMPE264 \n#Eduardo Hirata\n#Joshua Pena\n")

  #For Nikon pictures 'w1.jpg'
  # time=np.array([1.0/2500,1.0/1000,1.0/500,1.0/50,1.0/40,1.0/25],dtype='float32') # w
  time=np.array([1.0/320,1.0/200,1.0/80,1.0/50,1.0/25],dtype='float32') # p
  # time=np.array([1.0/1500,1.0/1000,1.0/750,1.0/500,1.0/350,1.0/250,1.0/125,1.0/45,1.0/30,1.0/20,1.0/15],dtype='float32') # j
  # time=np.array([1.0/1000,1.0/750,1.0/500,1.0/350,1.0/250,1.0/125,1.0/45],dtype='float32') # i
  # time=np.array([1.0/200,1.0/100,1.0/80,1.0/60,1.0/50,1.0/40,1.0/30],dtype='float32') # c

  color_calibration_images = get_color_calibration_images(time)

  for color_channel_index, color_channel_images in enumerate(tuple(color_calibration_images)):
    plt.figure()
    for n , img in enumerate(tuple(color_channel_images)):
        plt.subplot(subplot_col,subplot_row,n+1), plt.imshow(img,'gray')
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.7)
        plt.suptitle("Original Images")
        plt.title("Col="+ str(titles[color_channel_index])+" img=%d"%n)
        plt.savefig('./results/part_one/init_images-'+ str(titles[color_channel_index])+'.png', bbox_inches='tight')
    plt.close()

    plot_channel_calibration_histograms(color_channel_images, color_channel_index)

    #To get the central pixel, lets create a rectangular mask that takes the pixels in the middle
    mask = np.zeros(color_channel_images[0].shape[:2],np.uint8)
    center_w = int(round(color_channel_images[0].shape[0]/2))
    center_h = int(round(color_channel_images[0].shape[1]/2))
    # print(col[0].shape[0], col[0].shape[1])
    # print(center_w , center_h)

    ###############################Apply a Mask #####################################
    masked_img=[]
    hist_mask=[]
    crop_img=[]
    mu_img=[]

    mask[(center_w-off):(center_w+off), (center_h-off):(center_h+off)]=255  #Make the rest dark
    for img in color_channel_images:
      masked_img.append(cv2.bitwise_and(img,img, mask = mask))
      hist_mask.append(cv2.calcHist([img],[0],mask,[256],[0,256]))
      crop_img.append(img[(center_w-off):(center_w+off), (center_h-off):(center_h+off)])
      mu_img.append(np.mean(crop_img))

    plot_masked_images(masked_img, color_channel_index)

    plot_masked_images_histogram(hist_mask, color_channel_index)

    save_cropped_images(crop_img, color_channel_index)

    estimate_g_by_b(mu_img, time, color_channel_index)

    g = plot_log_b_for_t(mu_img, time, color_channel_index)

    b_a_estimation(mu_img, g, time, color_channel_index)

    color_g.append(np.float32(g))
    print('finished aquiring ' + titles[color_channel_index] + ' g')
  print('finished part one')


def part_two():
  original_images = []
  modified_images = []

  for image_index, pixel in enumerate(images):
    print(pixel)
    img = cv2.imread('images/'+str(pixel)+'.jpg')
    # img = cv2.resize(img, (408, 306)) # j, i
    img = cv2.resize(img, (576, 432)) # c

    original_images.append(img.copy())

    cv2.imwrite('./results/part_two/original_' + str(pixel) + '.png', img)

    create_histograms(img, './results/part_two/', str(pixel) + '_original', 0, 256, False)
    img = make_images_linear(img)
    create_histograms(img, './results/part_two/', str(pixel) + '_linear', 1, 25, True)

    cv2.imwrite('./results/part_two/new_' + str(pixel) + '_linear.png', img)

    if image_index != len(images) - 1:
      aValue = a_values[image_index]
      for height in range(img.shape[0]):
        for width in range(img.shape[1]):
          for color_index, color in enumerate(color_channels):
            img.itemset((height, width, color_index), img.item(height, width, color_index) / aValue)

      create_histograms(img, './results/part_two/', str(pixel) + '_modified', 2, 25, True)

      cv2.imwrite('./results/part_two/new_' + str(pixel) + '_modified.png', img)

    modified_images.append(img)
  
  return original_images, modified_images

def part_three(original_images, modified_images):
  hdr1 = copy_image_size(modified_images[0])
  hdr2 = copy_image_size(modified_images[0])

  for image_index, image in enumerate(tuple(original_images)):
    saturated_image = copy_image_size(image)
    for height in range(saturated_image.shape[0]):
      for width in range(saturated_image.shape[1]):
        if is_pixel_saturated(original_images[image_index], height, width):
          saturated_image[height, width] = [255, 255, 255]
        else:
          saturated_image[height, width] = [0, 0, 0]
    cv2.imwrite('./results/part_three/saturated_pixels_' + str(images[image_index % len(images)]) + '.png', saturated_image)

  pixel_from_image_array = []
  for image in modified_images:
    pixel_from_image_array.append(copy_image_size(image))

  pixel_from_together_hdr1 = copy_image_size(modified_images[0])
  pixel_from_together_hdr2 = copy_image_size(modified_images[0])

  for height in range(hdr1.shape[0]):
    for width in range(hdr1.shape[1]):
      for image_index, image in enumerate(tuple(modified_images)):
        if not is_pixel_saturated(original_images[image_index], height, width):
          hdr1[height, width] = image[height, width]

          pixel_from_image_array[image_index][height, width] = [255, 255, 255]
          pixel_from_together_hdr1[height, width] = from_color_array[image_index]
          break

      for color_index, color in enumerate(color_channels):
        averageValue = 0
        valuesUsed = 0
        for image_index, image in enumerate(tuple(modified_images)):
          if not is_pixel_saturated(original_images[image_index], height, width):
            averageValue += image.item(height, width, color_index)
            valuesUsed += 1
            pixel_from_together_hdr2.itemset((height, width, image_index), 255)
          else:
            pixel_from_together_hdr2.itemset((height, width, color_index), 0)

        if valuesUsed != 0:
          averageValue = averageValue / valuesUsed
        else:
          averageValue = 0
        hdr2.itemset((height, width, color_index), averageValue)

  create_histograms(hdr1, './results/part_three/', 'hdr1', 0, 256, True)
  create_histograms(hdr2, './results/part_three/', 'hdr2', 0, 256, True)
  cv2.imwrite('./results/part_three/hdr1.png', hdr1)
  cv2.imwrite('./results/part_three/hdr2.png', hdr2)

  cv2.imwrite('./results/part_three/pixel_from_together_hdr1.png', pixel_from_together_hdr1)
  cv2.imwrite('./results/part_three/pixel_from_together_hdr2.png', pixel_from_together_hdr2)
  for image_index, image in enumerate(tuple(pixel_from_image_array)):
    cv2.imwrite('./results/part_three/pixels_from_' + str(images[image_index]) + '.png', image)

  return hdr1, hdr2

def part_four(hdr1, hdr2, gamma, intensity, light_adapt, color_adapt):
  cv2.imwrite('./results/part_four/hdr1_original.png', hdr1)
  cv2.imwrite('./results/part_four/hdr2_original.png', hdr2)

  tonemap = cv2.createTonemapReinhard(gamma, intensity, light_adapt, color_adapt)
  tonemapHDR1 = tonemap.process(hdr1)
  tonemapHDR2 = tonemap.process(hdr2)

  hdr1_tonemap = np.clip(tonemapHDR1 * 255, 0, 255).astype('uint8')
  hdr2_tonemap = np.clip(tonemapHDR2 * 255, 0, 255).astype('uint8')
  cv2.imwrite('./results/part_four/hdr1_tonemap_g' + str(gamma) + '_i' + str(intensity) + '_la' + str(light_adapt) + '_ca' + str(color_adapt) + '.png', hdr1_tonemap)
  cv2.imwrite('./results/part_four/hdr2_tonemap_g' + str(gamma) + '_i' + str(intensity) + '_la' + str(light_adapt) + '_ca' + str(color_adapt) + '.png', hdr2_tonemap)

  cv2.imwrite('./results/part_four/hdr1_tonemap_v2_g' + str(gamma) + '_i' + str(intensity) + '_la' + str(light_adapt) + '_ca' + str(color_adapt) + '.png', tonemapHDR1 * 255)
  cv2.imwrite('./results/part_four/hdr2_tonemap_v2_g' + str(gamma) + '_i' + str(intensity) + '_la' + str(light_adapt) + '_ca' + str(color_adapt) + '.png', tonemapHDR2 * 255)

def main():
  part_one()

  original_images, modified_images = part_two()

  hdr1, hdr2 = part_three(original_images, modified_images)

  part_four(hdr1, hdr2, 3.5, -2.0, 0.25, 0.75)

  part_four(hdr1, hdr2, 3.5, 0.0, 0.0, 0.0)
  part_four(hdr1, hdr2, 3.5, -5.0, 0.0, 0.0)
  part_four(hdr1, hdr2, 3.5, 0.0, 0.25, 0.0)
  part_four(hdr1, hdr2, 3.5, 0.0, 0.0, 0.25)

main()
