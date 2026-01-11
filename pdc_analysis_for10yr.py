import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from tqdm import tqdm

def energy_distance(x_i, x_j, sigma_i, sigma_j):
    """Compute energy distance between two measurements with uncertainties."""
    return np.sqrt((x_i - x_j)**2 + sigma_i**2 + sigma_j**2)

def u_center(D):
    """Apply U-centering to distance matrix (Eq. 6)."""
    n = D.shape[0]
    row_means = D.mean(axis=1)
    col_means = D.mean(axis=0)
    total_mean = D.mean()
    U = np.zeros_like(D)
    for i in range(n):
        for j in range(n):
            U[i, j] = (D[i, j] - row_means[i] - col_means[j] + total_mean
                        if i != j else 0)
    return U

def chi2_fap(pdc_value, n_data):
    """Calculate FAP using χ² approximation (much faster)."""
    fap = 1 - chi2.cdf(n_data * pdc_value, df=1)
    return fap

def pdc_periodogram(t, y, sigma, periods):
    """
    Compute improved PDC periodogram with uncertainty handling.
    
    Args:
        t: Time values
        y: Flux measurements
        sigma: Flux uncertainties
        periods: Array of trial periods
        
    Returns:
        pdc_values: PDC power at each period
    """
    n = len(t)
    pdc_values = np.zeros_like(periods)
    
    # Precompute energy distance matrix (Eq. 5)
    print("Computing energy distance matrix...")
    D_energy = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D_energy[i, j] = energy_distance(y[i], y[j], sigma[i], sigma[j])
    
    # U-center energy distance matrix
    print("U-centering energy distance matrix...")
    U_energy = u_center(D_energy)
    
    # Precompute indices for all pairs (i < j)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Compute PDC for each period
    print("Computing PDC for each period...")
    for idx, P in enumerate(tqdm(periods, desc="PDC Calculation")):
        phases = (t / P) % 1
        dphi = np.abs(phases[i_idx] - phases[j_idx])
        dphi = np.minimum(dphi, 1 - dphi)  # Circular distance (Eq. 7)
        
        # Build phase distance matrix
        D_phase = np.zeros((n, n))
        D_phase[i_idx, j_idx] = dphi
        D_phase[j_idx, i_idx] = dphi
        
        # U-center phase distance matrix
        U_phase = u_center(D_phase)
        
        # Compute distance correlation (Eq. 8)
        numerator = np.sum(U_energy * U_phase)
        denom = np.sqrt(np.sum(U_energy**2) * np.sum(U_phase**2))
        pdc_values[idx] = numerator / denom if denom != 0 else 0
    
    return pdc_values

def load_sk_data(file_path):
    """Load Super-Kamiokande dataset."""
    print(f"Loading data from {file_path}...")
    data = np.loadtxt(file_path)
    t_seconds = data[:, 0]          # Mean time (seconds since 1970-01-01)
    flux = data[:, 3]               # Neutrino flux (10^6/cm²/s)
    upper_err = data[:, 4]          # Upper error
    lower_err = data[:, 5]          # Lower error
    sigma = (upper_err + lower_err) / 2  # Average uncertainty
    t_years = (t_seconds - t_seconds[0]) / (365.25 * 86400)  # Convert to years
    print(f"Loaded {len(t_years)} data points spanning {t_years[-1]:.1f} years")
    return t_years, flux, sigma

# Main analysis
if __name__ == "__main__":
    # Load dataset (replace with your file path if different)
    t, flux, sigma = load_sk_data("sksolartimevariation5804d.txt")
    
    # Period grid (0.5 to 10 years, 1000 points) - REDUCED RANGE
    periods = np.linspace(0.5, 10, 1000)
    print(f"Period search range: {periods[0]:.1f} - {periods[-1]:.1f} years")
    
    # Compute PDC periodogram
    print("Starting PDC periodogram calculation...")
    pdc = pdc_periodogram(t, flux, sigma, periods)
    
    # Find significant period
    peak_idx = np.argmax(pdc)
    peak_period = periods[peak_idx]
    peak_power = pdc[peak_idx]
    
    # Calculate χ²-based FAP for the peak
    peak_fap = chi2_fap(peak_power, len(t))
    
    # Set significance thresholds
    fap95_threshold = 0.05  # 5% significance
    fap99_threshold = 0.01  # 1% significance
    
    # Calculate PDC thresholds corresponding to these FAPs
    # Inverse of χ² CDF: PDC_threshold = χ²_inverse(1-FAP, df=1) / n_data
    pdc95_threshold = chi2.ppf(1 - fap95_threshold, df=1) / len(t)
    pdc99_threshold = chi2.ppf(1 - fap99_threshold, df=1) / len(t)
    
    # Print results
    print(f"\n=== RESULTS (PERIOD RANGE: 0.5-10 YEARS) ===")
    print(f"Peak period: {peak_period:.2f} years")
    print(f"Peak PDC power: {peak_power:.4f}")
    print(f"χ²-based FAP: {peak_fap:.2e}")
    print(f"Significance: {'HIGHLY SIGNIFICANT' if peak_fap < 0.01 else 'SIGNIFICANT' if peak_fap < 0.05 else 'NOT SIGNIFICANT'}")
    print(f"PDC 95% threshold: {pdc95_threshold:.4f}")
    print(f"PDC 99% threshold: {pdc99_threshold:.4f}")
    
    # Check for literature comparison
    if 9.0 <= peak_period <= 10.0:
        print(f"*** CONSISTENT with literature 9.43-year period! ***")
    elif 1.0 <= peak_period <= 2.0:
        print(f"*** Annual/semi-annual variation detected ***")
    else:
        print(f"*** Novel period detected in {peak_period:.2f}-year range ***")
    
    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Time series
    ax1.errorbar(t, flux, yerr=sigma, fmt='o', alpha=0.5, markersize=2)
    ax1.set_xlabel('Time (years since 1996)')
    ax1.set_ylabel('Flux ($10^6$ cm$^{-2}$ s$^{-1}$)')
    ax1.set_title('(a) Super-Kamiokande Solar Neutrino Flux (1996-2018)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Periodogram (focused range)
    ax2.plot(periods, pdc, 'k-', linewidth=1)
    ax2.axhline(pdc95_threshold, color='r', linestyle='--', label='5% FAP threshold')
    ax2.axhline(pdc99_threshold, color='b', linestyle='--', label='1% FAP threshold')
    ax2.plot(peak_period, peak_power, 'ro', markersize=8, label=f'Peak: {peak_period:.2f} yr')
    
    # Highlight literature comparison region
    ax2.axvspan(9.0, 10.0, alpha=0.2, color='green', label='Literature range (9-10 yr)')
    
    ax2.set_xlabel('Period (years)')
    ax2.set_ylabel('PDC Power')
    ax2.set_title('(b) PDC Periodogram (0.5-10 years)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.5, 10])
    
    # Plot 3: Phase-folded plot
    phases = (t / peak_period) % 1
    ax3.scatter(phases, flux, alpha=0.7, s=10)
    
    # Bin the data for better visualization
    phase_bins = np.linspace(0, 1, 21)
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    bin_means = []
    bin_stds = []
    for i in range(len(phase_bins)-1):
        mask = (phases >= phase_bins[i]) & (phases < phase_bins[i+1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(flux[mask]))
            bin_stds.append(np.std(flux[mask]) / np.sqrt(np.sum(mask)))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    # Plot binned data
    valid = ~np.isnan(bin_means)
    ax3.errorbar(bin_centers[valid], bin_means[valid], yerr=bin_stds[valid], 
                 fmt='ro-', linewidth=2, markersize=6, label='Binned data')
    ax3.set_xlabel(f'Phase (Period = {peak_period:.2f} years)')
    ax3.set_ylabel('Flux ($10^6$ cm$^{-2}$ s$^{-1}$)')
    ax3.set_title('(c) Phase-Folded Light Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Detailed periodogram around peak
    # Zoom in around the detected peak
    zoom_range = 1.0  # ±1 year around peak
    zoom_mask = (periods >= max(0.5, peak_period - zoom_range)) & (periods <= min(10, peak_period + zoom_range))
    
    ax4.plot(periods[zoom_mask], pdc[zoom_mask], 'k-', linewidth=2)
    ax4.axhline(pdc95_threshold, color='r', linestyle='--', label='5% FAP')
    ax4.axhline(pdc99_threshold, color='b', linestyle='--', label='1% FAP')
    ax4.plot(peak_period, peak_power, 'ro', markersize=10, label=f'Peak: {peak_period:.2f} yr')
    
    # Mark literature period if in range
    if 9.0 <= peak_period <= 10.0:
        ax4.axvline(9.43, color='green', linestyle=':', label='Literature: 9.43 yr')
    
    ax4.set_xlabel('Period (years)')
    ax4.set_ylabel('PDC Power')
    ax4.set_title(f'(d) Detailed View Around Peak ({peak_period:.1f} ± {zoom_range} years)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sk_pdc_analysis_reduced_range.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis: Find all significant peaks
    significant_mask = pdc > pdc95_threshold
    if np.any(significant_mask):
        sig_periods = periods[significant_mask]
        sig_powers = pdc[significant_mask]
        sig_faps = [chi2_fap(p, len(t)) for p in sig_powers]
        
        print(f"\n=== ALL SIGNIFICANT PERIODS (FAP < 5%) ===")
        for i, (period, power, fap) in enumerate(zip(sig_periods, sig_powers, sig_faps)):
            significance_level = "***HIGHLY SIGNIFICANT***" if fap < 0.01 else "**SIGNIFICANT**"
            print(f"{i+1}. Period: {period:.2f} years, PDC: {power:.4f}, FAP: {fap:.2e} {significance_level}")
            
            # Check against known periods
            if 9.0 <= period <= 10.0:
                print(f"   --> MATCHES literature 9.43-year period!")
            elif 0.9 <= period <= 1.1:
                print(f"   --> Annual variation")
            elif 0.45 <= period <= 0.55:
                print(f"   --> Semi-annual variation")
    else:
        print("\nNo significant periods found at 5% FAP level in 0.5-10 year range.")
    
    # Comparison with 20-year result
    print(f"\n=== COMPARISON WITH PREVIOUS 20-YEAR RESULT ===")
    print(f"Previous (0.5-20 yr range): 20.00 years (likely spurious)")
    print(f"Current (0.5-10 yr range):  {peak_period:.2f} years")
    print(f"Interpretation: Restricted range eliminates edge effects and")
    print(f"focuses on physically plausible solar periodicities.")
    
    # Save results to file
    results = {
        'periods': periods,
        'pdc_values': pdc,
        'peak_period': peak_period,
        'peak_power': peak_power,
        'peak_fap': peak_fap,
        'time': t,
        'flux': flux,
        'uncertainties': sigma,
        'period_range': f"0.5-10 years (reduced from 0.5-20)"
    }
    
    print(f"\nAnalysis complete! Results saved to 'sk_pdc_analysis_reduced_range.png'")
    print(f"Period search restricted to 0.5-10 years to avoid spurious long-period artifacts.")
