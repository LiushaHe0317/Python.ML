
import PyPDF2

with open('US_Declaration.pdf', mode = 'rb') as f1:
    
    # get page one content
    pdf_reader = PyPDF2.PdfFileReader(f1)
    page_one = pdf_reader.getPage(0)
    
    # add page one content into a new pdf file
    pdf_writer = PyPDF2.PdfFileWriter()
    pdf_writer.addPage(page_one)
    with open('MY BRAND NEW.pdf', mode = 'wb') as f2:
        pdf_output = pdf_writer.write(f2)
    
    with open('pdf_text.txt', mode = 'w') as f3:
        for each_page in range(pdf_reader.numPages):
            the_page = pdf_reader.getPage(each_page)
            f3.write(the_page.extractText())
            

        
