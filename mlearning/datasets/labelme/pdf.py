import os.path as osp
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

def create_pdf(title, subtitle, output_dir):
    c = canvas.Canvas(osp.join(output_dir, 'analysis.pdf'), pagesize=letter)
    
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(letter[0] / 2, letter[1] - 50, title)
    
    c.setFont("Helvetica", 18)
    c.drawCentredString(letter[0] / 2, letter[1] - 80, subtitle)
    
    ##########################################################################################
    content = "1. Number of Images in train/val dataset"
    c.drawText(get_text_object(c, content, 120))
    img_file = osp.join(output_dir, 'num_images_bar.png')
    img = Image.open(img_file)
    width, height = img.size
    ratio = height / float(width)
    aspect = 0.5
    c.drawInlineImage(img_file, 50, letter[1] - 350, width=letter[0]*aspect, height=letter[0]*ratio*aspect)

    img_file = osp.join(output_dir, 'num_images_pie.png')
    img = Image.open(img_file)
    width, height = img.size
    ratio = height / float(width)
    aspect = 0.4
    c.drawInlineImage(img_file, letter[0]/2 + 20, letter[1] - 380, width=letter[0]*aspect, height=letter[0]*ratio*aspect)

    ##########################################################################################
    content = "2. Number of Objects in train/val dataset"
    c.drawText(get_text_object(c, content, 410))
    img_file = osp.join(output_dir, 'train_pie_chart.png')
    img = Image.open(img_file)
    width, height = img.size
    ratio = height / float(width)
    aspect = 0.4
    c.drawInlineImage(img_file, 50, letter[1] - 700, width=letter[0]*aspect, height=letter[0]*ratio*aspect)

    img_file = osp.join(output_dir, 'val_pie_chart.png')
    img = Image.open(img_file)
    width, height = img.size
    ratio = height / float(width)
    aspect = 0.4
    c.drawInlineImage(img_file, letter[0]/2 + 20, letter[1] - 700, width=letter[0]*aspect, height=letter[0]*ratio*aspect)

    content = "The detailed information by each image is in the csv or html file."
    c.drawText(get_text_object(c, content, 710, 50))

    c.showPage()
    
    ##########################################################################################
    content = "3-1. Size of Objects"
    c.drawText(get_text_object(c, content, 50))
    img_file = osp.join(output_dir, 'objects_size_by_labels.png')
    img = Image.open(img_file)
    width, height = img.size
    ratio = height / float(width)
    aspect = 0.7
    c.drawInlineImage(img_file, 50, letter[1] - 500, width=letter[0]*aspect, height=letter[0]*ratio*aspect)

    c.showPage()
    c.save()


def get_text_object(c, content, loc_y, loc_x=50):
    c.setFont("Helvetica", 12)
    text_object = c.beginText(loc_x, letter[1] - loc_y)
    text_object.setFont("Helvetica", 12)
    
    for line in content.splitlines():
        text_object.textLine(line)
        
    return text_object