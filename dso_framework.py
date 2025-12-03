#!/usr/bin/env python3
"""
DSO Framework: Drag-Scale-Object Recovery for TEM Imaging
==========================================================

A geometric approach to image recovery using known reference structures
(e.g., graphene lattice) to measure and correct local distortion and blur.

API:
    PSC - P-Scale Coefficient: Reference geometry measurement
    WDC - W-Drag Coefficient: Local displacement and blur field
    GCO - G-Corrected Output: Recovered image

Pipeline:
    PSC(image, geometry) → reference positions
    WDC(expected, observed) → {Dx, Dy, spread} fields
    GCO(image, wdc, mode) → corrected output

Modes:
    'blur'       - Low distortion, high blur → Wiener + spread-guided refinement
    'distortion' - High distortion, low blur → Wiener then geometric unwarp
    'both'       - Significant both → Full correction pipeline

Reference:
    Garrett, J. (2025). DSO Framework for Geometric Image Recovery.

License: MIT
"""

import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import griddata


# =============================================================================
# PSC: P-Scale Coefficient
# =============================================================================

def PSC_graphene(image_shape, pixel_size_nm=0.02):
    """
    Generate expected graphene lattice positions (P-Scale Coefficient).
    
    Graphene has a hexagonal lattice with:
        - Lattice constant: a = 2.46 Å (0.246 nm)
        - C-C bond distance: 1.42 Å
        - Two atoms per unit cell
    
    Parameters
    ----------
    image_shape : tuple
        (H, W) image dimensions in pixels
    pixel_size_nm : float
        Pixel size in nanometers (default 0.02 nm for HRTEM)
    
    Returns
    -------
    positions : ndarray, shape (N, 2)
        Expected (x, y) positions of atoms in pixel coordinates
    metadata : dict
        Lattice parameters and FOV information
    """
    H, W = image_shape
    a_nm = 0.246  # Graphene lattice constant in nm
    
    # Lattice basis vectors
    a1 = np.array([a_nm, 0.0])
    a2 = np.array([a_nm / 2, a_nm * np.sqrt(3) / 2])
    
    # Field of view
    fov_nm = max(H, W) * pixel_size_nm
    max_n = int(fov_nm / a_nm) + 5
    
    # Generate hexagonal lattice with 2-atom basis
    coords = []
    for i in range(-max_n, max_n + 1):
        for j in range(-max_n, max_n + 1):
            r = i * a1 + j * a2
            coords.append(r)  # Atom A
            coords.append(r + np.array([0, a_nm / np.sqrt(3)]))  # Atom B
    
    coords = np.array(coords)
    coords -= coords.mean(axis=0)  # Center on origin
    
    # Keep positions within FOV
    fov_x = W * pixel_size_nm
    fov_y = H * pixel_size_nm
    margin = 0.9
    mask = (np.abs(coords[:, 0]) < fov_x / 2 * margin) & \
           (np.abs(coords[:, 1]) < fov_y / 2 * margin)
    coords = coords[mask]
    
    # Convert to pixel coordinates
    positions = coords / pixel_size_nm + np.array([W / 2, H / 2])
    
    metadata = {
        'lattice_constant_nm': a_nm,
        'pixel_size_nm': pixel_size_nm,
        'fov_nm': (fov_x, fov_y),
        'n_atoms': len(positions)
    }
    
    return positions, metadata


# =============================================================================
# WDC: W-Drag Coefficient
# =============================================================================

def WDC_measure(image, expected_positions, search_radius=6, dark_atoms=True):
    """
    Measure W-Drag Coefficients: local displacement and spread at each atom.
    
    At each expected atom position, find the observed centroid and measure
    the local spread (second moment), which indicates blur.
    
    Parameters
    ----------
    image : ndarray
        Input image (H, W)
    expected_positions : ndarray, shape (N, 2)
        Expected (x, y) positions from PSC
    search_radius : int
        Pixel radius to search around each expected position
    dark_atoms : bool
        True if atoms appear dark on bright background (typical TEM)
    
    Returns
    -------
    wdc : dict
        'observed': (N, 2) observed positions
        'displacement': (N, 2) displacement vectors (observed - expected)
        'spread': (N,) local spread (blur indicator) at each atom
        'Dx': (H, W) interpolated x-displacement field
        'Dy': (H, W) interpolated y-displacement field
        'spread_map': (H, W) interpolated spread field
        'mean_displacement': scalar, mean displacement magnitude
        'mean_spread': scalar, mean spread value
    """
    H, W = image.shape
    
    # Invert if atoms are dark
    work = 1.0 - image if dark_atoms else image.copy()
    
    observed = []
    spreads = []
    
    for x, y in expected_positions:
        x0, y0 = int(round(x)), int(round(y))
        r = search_radius
        x1, x2 = max(0, x0 - r), min(W, x0 + r + 1)
        y1, y2 = max(0, y0 - r), min(H, y0 + r + 1)
        
        if x1 >= x2 or y1 >= y2:
            observed.append((x, y))
            spreads.append(2.0)
            continue
        
        patch = work[y1:y2, x1:x2]
        patch = patch - patch.min()  # Local background subtraction
        total = patch.sum()
        
        if total <= 0:
            observed.append((x, y))
            spreads.append(2.0)
            continue
        
        # Centroid
        yy, xx = np.mgrid[y1:y2, x1:x2]
        cx = (patch * xx).sum() / total
        cy = (patch * yy).sum() / total
        
        # Second moment (spread)
        vx = (patch * (xx - cx) ** 2).sum() / total
        vy = (patch * (yy - cy) ** 2).sum() / total
        spread = np.sqrt((vx + vy) / 2)
        
        observed.append((cx, cy))
        spreads.append(spread)
    
    observed = np.array(observed)
    spreads = np.array(spreads)
    displacement = observed - expected_positions
    
    # Interpolate to full image grid
    xi, yi = np.meshgrid(np.arange(W), np.arange(H))
    
    Dx = griddata(expected_positions, displacement[:, 0], 
                  (xi, yi), method='linear', fill_value=0)
    Dy = griddata(expected_positions, displacement[:, 1], 
                  (xi, yi), method='linear', fill_value=0)
    spread_map = griddata(expected_positions, spreads, 
                          (xi, yi), method='linear', fill_value=np.mean(spreads))
    
    # Smooth to fill gaps
    Dx = gaussian_filter(np.nan_to_num(Dx, 0), 3)
    Dy = gaussian_filter(np.nan_to_num(Dy, 0), 3)
    spread_map = gaussian_filter(np.nan_to_num(spread_map, np.mean(spreads)), 3)
    
    mean_disp = np.mean(np.hypot(displacement[:, 0], displacement[:, 1]))
    mean_spread = np.mean(spreads)
    
    return {
        'observed': observed,
        'displacement': displacement,
        'spread': spreads,
        'Dx': Dx,
        'Dy': Dy,
        'spread_map': spread_map,
        'mean_displacement': mean_disp,
        'mean_spread': mean_spread
    }


# =============================================================================
# GCO: G-Corrected Output
# =============================================================================

def _wiener_deconvolution(image, sigma, K=None):
    """Wiener deconvolution with Gaussian PSF."""
    H, W = image.shape
    if K is None:
        K = 0.005 + 0.01 * sigma  # Adaptive regularization
    
    y, x = np.ogrid[-H//2:H//2, -W//2:W//2]
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()
    
    O = fft.fft2(image)
    P = fft.fft2(fft.fftshift(psf))
    G = np.conj(P) / (np.abs(P)**2 + K)
    
    return np.clip(np.real(fft.ifft2(G * O)), 0, 1)


def _find_optimal_wiener_sigma(image, sigma_range=(0.5, 2.0), n_steps=10):
    """Find Wiener sigma that maximizes high-frequency energy."""
    H, W = image.shape
    
    def hf_energy(im):
        F = np.abs(fft.fftshift(fft.fft2(im)))**2
        Y, X = np.ogrid[:H, :W]
        R = np.sqrt((X - W//2)**2 + (Y - H//2)**2)
        mask = R > min(H, W) / 6
        return F[mask].sum() / (F.sum() + 1e-12)
    
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_steps)
    best_sigma, best_hf = sigmas[0], 0
    
    for s in sigmas:
        w = _wiener_deconvolution(image, s)
        hf = hf_energy(w)
        if hf > best_hf:
            best_sigma, best_hf = s, hf
    
    return best_sigma


def _inverse_warp(image, Dx, Dy):
    """Apply inverse geometric warp."""
    H, W = image.shape
    xi, yi = np.meshgrid(np.arange(W), np.arange(H))
    coords = np.vstack([(yi + Dy).ravel(), (xi + Dx).ravel()])
    return map_coordinates(image, coords, order=1, mode='reflect').reshape(H, W)


def GCO(image, wdc, mode='auto'):
    """
    Generate G-Corrected Output using measured W-Drag Coefficients.
    
    Parameters
    ----------
    image : ndarray
        Input image (H, W), normalized to [0, 1]
    wdc : dict
        W-Drag Coefficients from WDC_measure()
    mode : str
        'auto' - Automatically select based on displacement vs spread
        'blur' - Optimize for blur removal (low distortion case)
        'distortion' - Optimize for geometric correction (high distortion case)
        'both' - Apply full correction pipeline
    
    Returns
    -------
    result : dict
        'output': corrected image
        'mode': mode used
        'wiener_sigma': Wiener sigma used (if applicable)
        'improvement': dict of metrics
    """
    H, W = image.shape
    mean_disp = wdc['mean_displacement']
    mean_spread = wdc['mean_spread']
    
    # Auto-select mode
    if mode == 'auto':
        if mean_disp < 1.0 and mean_spread > 2.0:
            mode = 'blur'
        elif mean_disp > 2.0 and mean_spread < 2.0:
            mode = 'distortion'
        else:
            mode = 'both'
    
    # Find optimal Wiener sigma
    optimal_sigma = _find_optimal_wiener_sigma(image)
    
    if mode == 'blur':
        # Low distortion, high blur: Wiener + spread-guided refinement
        wiener_result = _wiener_deconvolution(image, optimal_sigma)
        
        # Spread-guided local refinement
        spread_map = wdc['spread_map']
        sp_min, sp_max = spread_map.min(), spread_map.max()
        
        if sp_max > sp_min + 0.01:
            spread_norm = (spread_map - sp_min) / (sp_max - sp_min)
        else:
            spread_norm = np.zeros_like(spread_map)
        
        # Extra sharpening where spread is high
        blurred = gaussian_filter(wiener_result, 1.0)
        detail = wiener_result - blurred
        strength = spread_norm * 0.5
        output = np.clip(wiener_result + detail * strength, 0, 1)
        
    elif mode == 'distortion':
        # High distortion, low blur: Wiener then geometric unwarp
        wiener_result = _wiener_deconvolution(image, optimal_sigma)
        output = _inverse_warp(wiener_result, wdc['Dx'], wdc['Dy'])
        
    else:  # mode == 'both'
        # Full pipeline: unwarp + spatially-varying deblur
        unwarped = _inverse_warp(image, wdc['Dx'], wdc['Dy'])
        wiener_result = _wiener_deconvolution(unwarped, optimal_sigma)
        
        # Spread-guided refinement
        spread_map = wdc['spread_map']
        sp_min, sp_max = spread_map.min(), spread_map.max()
        
        if sp_max > sp_min + 0.01:
            spread_norm = (spread_map - sp_min) / (sp_max - sp_min)
        else:
            spread_norm = np.zeros_like(spread_map)
        
        blurred = gaussian_filter(wiener_result, 1.0)
        detail = wiener_result - blurred
        strength = spread_norm * 0.5
        output = np.clip(wiener_result + detail * strength, 0, 1)
    
    # Compute improvement metrics
    def hf_energy(im):
        F = np.abs(fft.fftshift(fft.fft2(im)))**2
        Y, X = np.ogrid[:H, :W]
        R = np.sqrt((X - W//2)**2 + (Y - H//2)**2)
        mask = R > min(H, W) / 6
        return F[mask].sum() / (F.sum() + 1e-12)
    
    def local_contrast(im):
        bl = gaussian_filter(im, 3)
        return np.mean(np.sqrt(gaussian_filter((im - bl)**2, 3)))
    
    hf_orig = hf_energy(image)
    hf_out = hf_energy(output)
    lc_orig = local_contrast(image)
    lc_out = local_contrast(output)
    
    return {
        'output': output,
        'mode': mode,
        'wiener_sigma': optimal_sigma,
        'improvement': {
            'hf_energy_original': hf_orig,
            'hf_energy_corrected': hf_out,
            'hf_improvement_pct': (hf_out - hf_orig) / hf_orig * 100,
            'contrast_original': lc_orig,
            'contrast_corrected': lc_out,
            'contrast_improvement_pct': (lc_out - lc_orig) / lc_orig * 100
        }
    }


# =============================================================================
# High-level API
# =============================================================================

def dso_recover(image, pixel_size_nm=0.02, material='graphene', mode='auto', 
                dark_atoms=True, verbose=True):
    """
    Full DSO recovery pipeline.
    
    Parameters
    ----------
    image : ndarray
        Input TEM image (H, W), will be normalized to [0, 1]
    pixel_size_nm : float
        Pixel size in nanometers
    material : str
        Reference material ('graphene' currently supported)
    mode : str
        Correction mode ('auto', 'blur', 'distortion', 'both')
    dark_atoms : bool
        True if atoms appear dark on bright background
    verbose : bool
        Print progress and results
    
    Returns
    -------
    result : dict
        'output': corrected image
        'psc': P-Scale Coefficient data
        'wdc': W-Drag Coefficient data
        'gco': G-Corrected Output data
    """
    # Normalize image
    image = np.asarray(image, dtype=np.float64)
    image = (image - image.min()) / (image.max() - image.min() + 1e-10)
    
    if verbose:
        print("=" * 60)
        print("DSO Framework: Geometric Image Recovery")
        print("=" * 60)
    
    # Step 1: PSC - Generate reference geometry
    if verbose:
        print(f"\n[PSC] Generating {material} reference geometry...")
    
    if material == 'graphene':
        positions, psc_meta = PSC_graphene(image.shape, pixel_size_nm)
    else:
        raise ValueError(f"Material '{material}' not supported")
    
    if verbose:
        print(f"      {psc_meta['n_atoms']} reference positions")
        print(f"      FOV: {psc_meta['fov_nm'][0]:.1f} x {psc_meta['fov_nm'][1]:.1f} nm")
    
    # Step 2: WDC - Measure drag field
    if verbose:
        print(f"\n[WDC] Measuring drag coefficients...")
    
    wdc = WDC_measure(image, positions, dark_atoms=dark_atoms)
    
    if verbose:
        print(f"      Mean displacement: {wdc['mean_displacement']:.2f} px")
        print(f"      Mean spread: {wdc['mean_spread']:.2f} px")
    
    # Step 3: GCO - Generate corrected output
    if verbose:
        if mode == 'auto':
            print(f"\n[GCO] Auto-selecting correction mode...")
        else:
            print(f"\n[GCO] Applying '{mode}' correction...")
    
    gco = GCO(image, wdc, mode=mode)
    
    if verbose:
        print(f"      Mode: {gco['mode']}")
        print(f"      Wiener σ: {gco['wiener_sigma']:.2f}")
        print(f"\n[Results]")
        print(f"      HF Energy: {gco['improvement']['hf_improvement_pct']:+.1f}%")
        print(f"      Contrast:  {gco['improvement']['contrast_improvement_pct']:+.1f}%")
        print("=" * 60)
    
    return {
        'output': gco['output'],
        'original': image,
        'psc': {'positions': positions, 'metadata': psc_meta},
        'wdc': wdc,
        'gco': gco
    }


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    print("\nDSO Framework - Example on Graphene TEM\n")
    
    # Check for test image
    test_paths = [
        '/mnt/user-data/uploads/3_5_2_TEM_Graphene.tif',
        '/mnt/user-data/uploads/3_5_7_TEM_Graphene.tif'
    ]
    
    test_image = None
    for p in test_paths:
        if Path(p).exists():
            from PIL import Image
            test_image = np.array(Image.open(p), dtype=np.float64)
            print(f"Loaded: {p}")
            print(f"Size: {test_image.shape}")
            break
    
    if test_image is None:
        # Generate synthetic test image
        print("No test image found. Generating synthetic data...")
        np.random.seed(42)
        H, W = 256, 256
        test_image = np.random.rand(H, W) * 0.1 + 0.5
    
    # Crop center if large
    if test_image.shape[0] > 512:
        h, w = test_image.shape
        c = 256
        test_image = test_image[h//2-c//2:h//2+c//2, w//2-c//2:w//2+c//2]
        print(f"Cropped to: {test_image.shape}")
    
    # Run DSO
    result = dso_recover(
        test_image,
        pixel_size_nm=0.02,
        material='graphene',
        mode='auto',
        verbose=True
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(result['original'], cmap='gray')
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(result['wdc']['spread_map'], cmap='viridis')
    axes[1].set_title('WDC: Spread Map', fontsize=14)
    axes[1].axis('off')
    
    imp = result['gco']['improvement']['hf_improvement_pct']
    axes[2].imshow(result['output'], cmap='gray')
    axes[2].set_title(f'GCO: Corrected ({imp:+.1f}%)', fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle('DSO Framework: PSC → WDC → GCO', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = '/mnt/user-data/outputs/dso_framework_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_path}")
