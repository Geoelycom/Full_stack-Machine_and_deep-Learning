#!/usr/bin/env python3
"""
DLABS Installation Verification Script

This script tests that all major components of the DLABS deep learning
codebase are properly installed and working.
"""

import sys
import importlib
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

def test_import(module_name: str, display_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    if display_name is None:
        display_name = module_name
    
    try:
        importlib.import_module(module_name)
        return True, f"✅ {display_name}"
    except ImportError as e:
        return False, f"❌ {display_name} - {str(e)}"
    except Exception as e:
        return False, f"⚠️  {display_name} - {str(e)}"

def test_pytorch_functionality() -> Tuple[bool, str]:
    """Test basic PyTorch functionality."""
    try:
        import torch
        
        # Test tensor creation
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        
        # Test CUDA availability (if available)
        cuda_info = f"CUDA available: {torch.cuda.is_available()}"
        if torch.cuda.is_available():
            cuda_info += f" (Device: {torch.cuda.get_device_name(0)})"
        
        return True, f"✅ PyTorch functionality - {cuda_info}"
    except Exception as e:
        return False, f"❌ PyTorch functionality - {str(e)}"

def test_computer_vision() -> Tuple[bool, str]:
    """Test computer vision libraries."""
    try:
        import cv2
        import PIL
        from torchvision import transforms
        
        # Test basic CV operations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        return True, f"✅ Computer Vision modules (OpenCV {cv2.__version__}, PIL {PIL.__version__})"
    except Exception as e:
        return False, f"❌ Computer Vision modules - {str(e)}"

def test_nlp() -> Tuple[bool, str]:
    """Test NLP libraries."""
    try:
        from transformers import pipeline
        
        # Test basic NLP functionality
        # Note: This doesn't download models, just tests import
        return True, "✅ NLP modules (Transformers ready)"
    except Exception as e:
        return False, f"❌ NLP modules - {str(e)}"

def test_ocr() -> Tuple[bool, str]:
    """Test OCR libraries."""
    try:
        import easyocr
        # Note: We don't initialize EasyOCR here as it downloads models
        return True, "✅ OCR modules (EasyOCR ready)"
    except Exception as e:
        return False, f"❌ OCR modules - {str(e)}"

def run_all_tests() -> Dict[str, List[Tuple[bool, str]]]:
    """Run all installation tests."""
    
    results = {
        "Core Dependencies": [],
        "Deep Learning": [],
        "Computer Vision": [],
        "NLP": [],
        "OCR": [],
        "Production": [],
        "Development": []
    }
    
    # Core dependencies
    core_modules = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn")
    ]
    
    for module, name in core_modules:
        results["Core Dependencies"].append(test_import(module, name))
    
    # Deep Learning
    results["Deep Learning"].append(test_import("torch", "PyTorch"))
    results["Deep Learning"].append(test_pytorch_functionality())
    results["Deep Learning"].append(test_import("torchvision", "TorchVision"))
    
    # Computer Vision
    results["Computer Vision"].append(test_computer_vision())
    results["Computer Vision"].append(test_import("timm", "TIMM"))
    results["Computer Vision"].append(test_import("albumentations", "Albumentations"))
    
    # NLP
    results["NLP"].append(test_nlp())
    results["NLP"].append(test_import("datasets", "Datasets"))
    
    # OCR
    results["OCR"].append(test_ocr())
    
    # Production
    production_modules = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic")
    ]
    
    for module, name in production_modules:
        results["Production"].append(test_import(module, name))
    
    # Development
    dev_modules = [
        ("jupyter", "Jupyter"),
        ("pytest", "Pytest"),
        ("black", "Black"),
        ("rich", "Rich")
    ]
    
    for module, name in dev_modules:
        results["Development"].append(test_import(module, name))
    
    return results

def display_results(results: Dict[str, List[Tuple[bool, str]]]):
    """Display test results in a formatted table."""
    
    table = Table(title="DLABS Installation Test Results", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    all_passed = True
    
    for category, tests in results.items():
        category_passed = all(passed for passed, _ in tests)
        all_passed = all_passed and category_passed
        
        status = "✅ PASS" if category_passed else "❌ FAIL"
        details = "\n".join(msg for _, msg in tests)
        
        table.add_row(category, status, details)
    
    console.print(table)
    
    # Summary
    if all_passed:
        console.print(Panel.fit(
            "🎉 [bold green]All tests passed! DLABS is ready to use.[/bold green]\n\n"
            "Next steps:\n"
            "1. Explore the examples in each module directory\n"
            "2. Check out the Jupyter notebooks in experiments/notebooks/\n"
            "3. Start building your deep learning projects!",
            title="Success",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "⚠️  [bold yellow]Some tests failed.[/bold yellow]\n\n"
            "Please check the installation instructions in README.md\n"
            "and ensure all dependencies are properly installed.",
            title="Warning",
            border_style="yellow"
        ))
    
    return all_passed

def main():
    """Main function to run all tests."""
    console.print(Panel.fit(
        "[bold blue]DLABS - Deep Learning Laboratory[/bold blue]\n"
        "Installation Verification Script\n\n"
        "Testing all components...",
        title="Welcome",
        border_style="blue"
    ))
    
    console.print("\n[bold]Running installation tests...[/bold]\n")
    
    results = run_all_tests()
    success = display_results(results)
    
    # System information
    console.print(f"\n[bold]System Information:[/bold]")
    console.print(f"Python version: {sys.version}")
    
    try:
        import torch
        console.print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            console.print(f"CUDA version: {torch.version.cuda}")
            console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            console.print("CUDA: Not available (CPU only)")
    except ImportError:
        console.print("PyTorch: Not installed")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
