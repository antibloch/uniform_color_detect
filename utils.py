import cv2
import os

def open_video(pth):
    cap = cv2.VideoCapture(pth)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")

    # Read until video is completed
    while(cap.isOpened()):
        
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        # Display the resulting frame
            cv2.imshow('Frame', frame)
            
        # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    cap.release()
    cv2.destroyAllWindows()




def save_video(img_pth, fps=24):
   img_list=os.listdir(img_pth)
   img_list.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
   img_list = [os.path.join(img_pth, img) for img in img_list]
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   
   # Read first image to get dimensions
   first_img = cv2.imread(img_list[0])
   height, width = first_img.shape[:2]
   
   # Create video writer object
   out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
   
   # Write each image to video
   for img_path in img_list:
       frame = cv2.imread(img_path)
       out.write(frame)
   
   # Release video writer
   out.release()