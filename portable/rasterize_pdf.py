"""PDFium implementation of the pdftoppm flags used by the frozen driver.

Only figure rasterisation changes; numerical calculations are unaffected.
The wrapper deliberately rejects flags outside the required interface.
"""
import argparse
import math
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
    formats = p.add_mutually_exclusive_group(required=True)
    formats.add_argument('-png', action='store_true')
    formats.add_argument('-jpeg', action='store_true')
    p.add_argument('-jpegopt')
    p.add_argument('pdf', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    if not args.singlefile or args.f < 1 or not math.isfinite(args.r) or args.r <= 0:
        p.error('This figure interface requires -singlefile, a valid page and positive finite resolution.')
    save_options = {'dpi': (args.r, args.r)}
    if args.jpegopt:
        if not args.jpeg:
            p.error('-jpegopt requires -jpeg.')
        for option in args.jpegopt.split(','):
            key, separator, value = option.partition('=')
            if separator and key == 'quality' and value.isdigit() and 0 <= int(value) <= 100:
                save_options[key] = int(value)
            elif separator and key in ('optimize', 'progressive') and value in ('y', 'n'):
                save_options[key] = value == 'y'
            else:
                p.error('Unsupported JPEG option: ' + option)
    with pdfium.PdfDocument(args.pdf) as document:
        if args.f > len(document):
            p.error('Requested PDF page does not exist.')
        page = document[args.f - 1]
        bitmap = page.render(scale=args.r / 72)
        try:
            output = Path(str(args.output) + ('.jpg' if args.jpeg else '.png'))
            output.parent.mkdir(parents=True, exist_ok=True)
            picture = bitmap.to_pil()
            if args.jpeg:
                picture = picture.convert('RGB')
            picture.save(output, format='JPEG' if args.jpeg else 'PNG', **save_options)
        finally:
            bitmap.close(); page.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
