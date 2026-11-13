"""PDFium implementation of the pdftoppm flags used by the frozen driver.

Only figure rasterisation changes; numerical calculations are unaffected.
The wrapper deliberately rejects flags outside the required interface.
"""
import argparse
from pathlib import Path
import sys


def main():
    import pypdfium2 as pdfium
    if sys.argv[1:] == ['-v']:
        from pypdfium2.version import PYPDFIUM_INFO, PDFIUM_INFO
        print(f'PISCO figure renderer: pypdfium2 {PYPDFIUM_INFO}, PDFium {PDFIUM_INFO}')
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-f', type=int, default=1)
    p.add_argument('-singlefile', action='store_true')
    p.add_argument('-r', type=float, default=150)
    p.add_argument('-png', action='store_true')
    p.add_argument('pdf', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    if not args.singlefile or not args.png or args.f < 1 or args.r <= 0:
        p.error('This figure interface requires -singlefile -png, a valid page and positive resolution.')
    with pdfium.PdfDocument(args.pdf) as document:
        if args.f > len(document):
            p.error('Requested PDF page does not exist.')
        page = document[args.f - 1]
        bitmap = page.render(scale=args.r / 72)
        output = Path(str(args.output) + '.png')
        output.parent.mkdir(parents=True, exist_ok=True)
        bitmap.to_pil().save(output, format='PNG', dpi=(args.r, args.r))
        bitmap.close(); page.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
