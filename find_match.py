import recognizer
import os


#=============  Matching each images in test folder with the known faces ===============

# Get path to all the test images
# Filtering on .jpg extension - so this will only work with JPEG images ending with .jpg
def match_faces():
    test_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('test/'))

    # Get full paths to test images
    paths_to_test_images = ['test/' + x for x in test_filenames]
    res_path = 'recognized_faces/'
    # Get list of names of people by eliminating the .JPG extension from image filenames
    names = [x[:-4] for x in recognizer.image_filenames]

    # Iterate over test images to find match one by one
    for path_to_image in paths_to_test_images:
        # Get face encodings from the test image
        face_encodings_in_image = recognizer.get_face_encodings(path_to_image)
        # Make sure there's exactly one face in the image
        if len(face_encodings_in_image) != 1:
            print()
            print()
            print("Please change image: " + path_to_image + " - it has " + str(len
                        (face_encodings_in_image)) + " faces; it can only have one")
            print()
            print()
            continue

        # Find match for the face encoding found in this test image
        match = recognizer.find_match(recognizer.face_encodings, names, face_encodings_in_image[0])

        # Print the path of test image and the corresponding match
        print(path_to_image +"==> \t\t\t" + match)
        #Write result to result path
        #cv2.imwrite(match)

match_faces()