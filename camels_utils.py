import numpy as np

def fdc_metrics(sim, obs, high_frac=0.02, low_frac=0.3, p1=0.2, p2=0.7):
    """
    Compute Flow Duration Curve (FDC) metrics:
      - FHV: high-segment volume bias (%)
      - FMS: mid-segment slope ratio (%)
      - FLV: low-segment volume bias (%)
    
    Reference:
        Yilmaz, K. K., Gupta, H. V., & Wagener, T. (2008).
        A process‐based diagnostic approach to   model evaluation.
        J. Hydrometeorology, 9(1), 13–28.

    Parameters
    ----------
    sim : array-like
        Simulated (predicted) flow series.
    obs : array-like
        Observed flow series (same length as sim).
    high_frac : float, optional
        Fraction of top flows used for high segment (default 0.02 = top 2%).
    low_frac : float, optional
        Fraction of lowest flows used for low segment (default 0.3 = bottom 30%).
    p1, p2 : float, optional
        Exceedance probabilities for slope calculation (default 0.2 and 0.7).

    Returns
    -------
    metrics : dict
        Dictionary with keys:
        {'FHV': float, 'FMS': float, 'FLV': float}
    """
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if sim.shape != obs.shape:
        raise ValueError("sim and obs must have same length.")
    
    # remove NaNs
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    sim, obs = sim[mask], obs[mask]
    n = len(obs)
    if n == 0:
        raise Warning("time series too short for FDC metrics.")
        return 0, 0, 0
    
    # sort the array in descending order (FDC)
    obs_sorted = np.sort(obs)[::-1]
    sim_sorted = np.sort(sim)[::-1]
    
    # --- FHV: High-flow bias ---
    nH = int(np.ceil(high_frac * n))
    fhv = 100.0 * np.sum(sim_sorted[:nH] - obs_sorted[:nH]) / np.sum(obs_sorted[:nH])


    # --- FLV: Low-flow bias ---
    nL = int(np.floor((1 - low_frac) * n))
    flv = 100.0 * np.sum((sim_sorted[nL:] - obs_sorted[nL:])) / (np.sum(obs_sorted[nL:]) + 0.0001)

    # --- FMS: Mid-segment slope ratio ---
    exceed_probs = (np.arange(1, n+1)) / (n + 1.0)

    def flow_at(p, q_sorted):
        # interpolate discharge at exceedance probability p
        return np.interp(p, exceed_probs, q_sorted[::-1])  # need ascending order
    qs_p1, qs_p2 = flow_at(p1, sim_sorted), flow_at(p2, sim_sorted)
    qo_p1, qo_p2 = flow_at(p1, obs_sorted), flow_at(p2, obs_sorted)
    
    slope_sim = ((qs_p2) - (qs_p1)) / (p2 - p1)
    slope_obs = ((qo_p2) - (qo_p1)) / (p2 - p1)

    fms = 100.0 * (slope_sim / slope_obs - 1.0)
        
    return fhv, fms, flv

def printLatexTableRow(model_type, timestep=None, **kwargs):

    NSE = kwargs['NSE']
    KGE = kwargs['KGE']
    FHV = kwargs['FHV']
    FLV = kwargs['FLV']

    if not timestep is None:
        return f"""{model_type} & \
          {np.median(NSE[:,timestep]):4.3f} / {np.mean(NSE[:,timestep]):4.3f} & \
          {np.median(KGE[:,timestep]):4.3f} / {np.mean(KGE[:,timestep]):4.3f} & \
          {np.median(FHV[:,timestep]):6.3f} / {np.mean(FHV[:,timestep]):6.3f}  & \
          {np.median(FLV[:,timestep]):6.3f} / {np.mean(FLV[:,timestep]):6.3f} \\\\ \n"""
    else:
        return f"""{model_type} & \
          {np.median(NSE):4.3f} / {np.mean(NSE):4.3f} & \
          {np.median(KGE):4.3f} / {np.mean(KGE):4.3f} & \
          {np.median(FHV):6.3f} / {np.mean(FHV):6.3f} & \
          {np.median(FLV):6.3f} / {np.mean(FLV):6.3f} \\\\ \n"""

# ==== Example usage ====
if __name__ == "__main__":
    np.random.seed(42)
    obs = np.random.gamma(4, 15, 365)
    sim = obs * np.random.normal(1.1, 0.2, 365)  # slightly high bias
    metrics = fdc_metrics(sim, obs)
    print(metrics)
