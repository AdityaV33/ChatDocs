import os
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import time

def fetch_arxiv_pdfs(max_results=80, output_dir='evaluation/dataset/secondary_pdfs'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Query arXiv for cs.AI and cs.IR papers
    url = f'http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.IR&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
    
    print(f"Querying arXiv: {url}")
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return
        
    root = ET.fromstring(xml_data)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    
    entries = root.findall('atom:entry', namespace)
    print(f"Found {len(entries)} papers.")
    
    count = 0
    for entry in entries:
        title = entry.find('atom:title', namespace).text.replace('\n', ' ').strip()
        pdf_url = None
        for link in entry.findall('atom:link', namespace):
            if link.attrib.get('title') == 'pdf':
                pdf_url = link.attrib.get('href')
                break
                
        if pdf_url:
            pdf_url = pdf_url.replace('http://', 'https://')
            if not pdf_url.endswith('.pdf'):
                pdf_url += '.pdf'
                
            pdf_id = pdf_url.split('/')[-1].replace('.pdf', '')
            out_path = os.path.join(output_dir, f"{pdf_id}.pdf")
            
            if not os.path.exists(out_path):
                print(f"Downloading [{count+1}/{max_results}]: {title[:50]}... -> {out_path}")
                try:
                    urllib.request.urlretrieve(pdf_url, out_path)
                    time.sleep(3)  # Be nice to arXiv
                    count += 1
                except Exception as e:
                    print(f"  Failed: {e}")
            else:
                print(f"Skipping (exists): {pdf_id}")
                count += 1

if __name__ == "__main__":
    fetch_arxiv_pdfs(80)
