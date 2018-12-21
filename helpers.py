# IMPORT PACKAGES
import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------------- #

def print_documentation(argument : 'class, function, etc.'):
    """\n---FUNCTION DOCUMENTATION---

Function Name : print_documentation
This will print the Documentation for a given argument like a function, class, etc.

---END---\n"""
    
    try:
        print(argument.__doc__)
    except:
        print("\n!!!\tNO DOCUMENTATION FOUND\t!!!\n")
        
    return

# ----------------------------------------------------------------------------------------------------------------------------- #

def print_annotations(argument : 'class, function, etc.'):
    """\n---FUNCTION DOCUMENTATION---

Function Name : print_annotations
This will print the Annotations for a given argument like a function, class, etc.

---END---\n"""
    
    try:
        print("\n---ANNOTATIONS---\n\n",argument," : ",argument.__annotations__,"\n\n---END---\n")
    except :
        print("\n!!!\tNO ANNOTATIONS FOUND\t!!!\n")
    return

# ----------------------------------------------------------------------------------------------------------------------------- #

def segregate(images : list) -> dict :
    """\n---FUNCTION DOCUMENTATION---

Function Name : segregate

This will segragate the images as per the same labels and consolidate them into separate groups

---END---\n"""
    
    segregated_images = {}
    
    for image in images:
        label = image[1]

        if not label in segregated_images:
            segregated_images[label] = [image[0]]
        else:
            segregated_images[label].append(image[0])

    return segregated_images         

# ----------------------------------------------------------------------------------------------------------------------------- #

def show(image_no : '1 for 1st Image', images : list) -> 'None or Print':
    """\n---FUNCTION DOCUMENTATION---

Function Name : show
This will display the following details about an Image:
1. It's Shape.
2. Label associated to it.
3. The Image.

---END---\n"""
    
    error = None
    
    try:
        index = image_no-1
        total_images = len(images)
        
        selected_image = images[index][0]
        selected_image_label = images[index][1]

        print("\nDetails of Image "+str(index+1)+" / "+str(len(images))+" :\n")
        print("1. Shape : ",np.shape(selected_image))
        print("2. Label : ",selected_image_label,"\n")
        plt.imshow(selected_image)
        plt.show()
        
    except IndexError:
        print("!!! WARNING !!!\tImage no. should be between "+str(1)+" and "+str(total_images)+"\n")
        error = "Try Again"
        
    return error

# ----------------------------------------------------------------------------------------------------------------------------- #

def show_label_specific(images : dict):
    """\n---FUNCTION DOCUMENTATION---

Function Name : show_label_specific

Arguments :
1. Images segregated as per labels

Inputs & Outputs :
1. Presents the user with the available options of Labels
2. Asks the user to input the image no.
3. Outputs the details of the image.

Returns :
1. None

---END---\n"""
    
    options = {i+1 : key for i,key in enumerate(images.keys())}
    
    print("Choose from the following options:")
    for option in options.keys():
        print(" ",option,"for",options[option])
    
    while True:
        try:
            choice = int(input("\nEnter your choice : "))
            if not choice in options:
                raise ValueError
            break
                
        except ValueError:
            print("!!! WARNING !!!\tInvalid Option")
    
    label = options[choice]
    label_images = [[label_image,label] for label_image in images[label]]
    total_label_images = len(label_images)
    
    print("\n--- Total  %s '%s' images were found ---\n"%(total_label_images,label))
    while True:
        try:
            image_no = int(input("Enter '"+str(label)+"' - image no. between "+str(1)+" and "+str(total_label_images)+" : "))
            #print("\n")
            if show(image_no,label_images) != None :
                continue
            break
        except (ValueError):
            print("!!! WARNING !!!\tInvalid Option\n")
            
    return    

# ----------------------------------------------------------------------------------------------------------------------------- #

def plot_images(images:list,grid:tuple, size:tuple, image_titles:'`None`, `1` or a `list()`' = None,cmap_type:list = None, axis = 0,save_plot = 0,filepath = None,dpi=75,pad_inches=0.2,title_font=16):
    """
    Generates subplots and optionally saves them as well.
    """
    total_images = len(images)
    
    nrows = grid[0]
    ncols = grid[1]
    
    width = size[0]
    height = size[1]
    
    if image_titles == None:
        pass
    elif image_titles == 1:
        image_titles = ["Image #"+str(i+1) for i in range(total_images)]
    elif type(image_titles) == list:
        if len(image_titles) != total_images:
            raise ValueError("`image_titles` should have equal number of values as `images`")
    else: 
        raise ValueError("`image_titles` has to be either `None`, `1` or a `list()`")
    
   
    if cmap_type == None:
        cmap_type = [None for i in range(total_images)]      
    elif type(cmap_type) == list:
        if len(cmap_type) != total_images:
            raise ValueError("`cmap_type` should have equal number of values as `images`")
    else: 
        raise ValueError("`cmap_type` has to be either `None` or a `list()`")
        
    fig,axes = plt.subplots(nrows, ncols, figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.tight_layout()
    
    if axis == 0:
        for ax in axes.flatten():
            ax.axis('off')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
    
    if nrows == 1:
        for i in range(ncols):
            axes[i].imshow(images[i],cmap = cmap_type[i])
            if image_titles != None:
                axes[i].set_title(image_titles[i],fontsize=title_font)
    
    elif ncols == 1:
        for i in range(nrows):
            axes[i].imshow(images[i],cmap = cmap_type[i])
            if image_titles != None:
                axes[i].set_title(image_titles[i],fontsize=title_font)
                
    else:
        count = 0
        for i in range(nrows):
            for j in range(ncols):

                if count < total_images:
                    axes[i,j].imshow(images[count],cmap = cmap_type[count])
                    if image_titles != None:
                        axes[i,j].set_title(image_titles[count],fontsize=title_font)
                else:
                    break

                count+=1
    
    if save_plot == 1:
        fig.savefig(filepath,dpi=dpi,bbox_inches='tight',pad_inches=pad_inches)
    elif save_plot == 2:
        try:
            return fig
        finally:
            plt.close(fig)
    else:
        plt.show()
    
# ----------------------------------------------------------------------------------------------------------------------------- #