"""
Utilities for file processing, including:
1. File type detection
2. Text extraction from various file formats
3. File validation
"""
import os
from typing import Optional
import magic
import PyPDF2
import docx
from fastapi import HTTPException

# Supported file types and their extensions
SUPPORTED_EXTENSIONS = {
    '.txt': 'text/plain',
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.md': 'text/markdown',
}

def get_file_type(filename: str) -> str:
    """
    Get the file extension from a filename.
    Raises HTTPException if file type is not supported.
    """
    if not filename or '.' not in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported types are: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )
    
    return ext

def get_mime_type(file_path: str) -> str:
    """
    Get MIME type of a file using python-magic.
    """
    return magic.from_file(file_path, mime=True)

async def extract_text_from_file(file_path: str) -> str:
    """
    Extract text content from various file types.
    Supports: txt, pdf, docx, doc, md
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    mime_type = get_mime_type(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if mime_type == 'text/plain' or ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif mime_type == 'application/pdf' or ext == '.pdf':
            text = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)

        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or ext == '.docx':
            doc = docx.Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported MIME type: {mime_type}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from file: {str(e)}"
        ) 