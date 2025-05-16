# lognormality_app
#--------------------------------------------------------------------
#
# Demo app showing the concept of lognormality,
# i.e. multiplication of random indipendent variables is lognormal
# 
# to run: 
#
# $ streamlit run lognormality_app.py
#--------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
from matplotlib import colormaps as mplcmap
import streamlit as st

#--------------------------------------------------------------------
@st.cache_data
def cached_mult2dist(dist_name, mu, sigma, symm, nsamples ):
    return mult2dist(dist_name, mu, sigma, symm, nsamples)

def mult2dist(dist_name, mu, sigma, symm, nsamples):

    # get function for chosen distribution
    distribution = getattr(scipy.stats, dist_name)

    prod_samples = None  

    fig, ax = plt.subplots(nrows=2, constrained_layout=True)

    # loop over number of disttributions chosen by user
    for i in range(len(mu)):
        if dist_name == 'triang':
            dd = distribution(c=symm[i], loc=mu[i], scale=sigma[i])
        elif dist_name == 'norm':
            dd = distribution(loc=mu[i], scale=sigma[i])
        elif dist_name == 'uniform':
            dd = distribution(loc=mu[i], scale=sigma[i]) # uniform dist from loc to loc+scale

        dist_samples = dd.rvs(size=nsamples)

        # Generate samples and ensure we have nsamples positive values
        dist_samples = []
        while len(dist_samples) < nsamples:
            new_samples = dd.rvs(size=nsamples - len(dist_samples))
            dist_samples.extend(new_samples[new_samples > 0])
        dist_samples = np.array(dist_samples[:nsamples])  # Trim to exact size

        # cumulative multiplication of the distributions
        if prod_samples is None:
            prod_samples = dist_samples
        else:
            prod_samples = prod_samples * dist_samples

        # plot histogram of the distribution        
        ax[0].hist(dist_samples, alpha=0.5, density=True, color=colors[i])
        # plot the distribution pdf
        xx = np.linspace(dist_samples.min(), dist_samples.max())
        ax[0].plot(xx, dd.pdf(xx), color=colors[i])
        
    ax[1].hist(prod_samples, bins=50, alpha=0.5, density=True, color='k')

    for aa in ax:
        aa.spines['right'].set_visible(False)
        aa.spines['top'].set_visible(False)
        for spine in aa.spines.values():
            spine.set_linewidth(2)
        aa.yaxis.set_visible(False)
    
    return fig, ax, prod_samples


def update_lognormal_fit(ax, prod_samples, show_fit):
    # Remove existing lognormal fit line if it exists
    for line in ax.lines:
        line.remove()
    
    if show_fit and len(prod_samples) > 0:
        try:
            params = scipy.stats.lognorm.fit(prod_samples, floc=0)
            estim = scipy.stats.lognorm(*params)
            prod_x = np.linspace(estim.ppf(0.01), estim.ppf(0.99))
            ax.plot(prod_x, estim.pdf(prod_x), color='r', linewidth=2)
        except Exception as e:
            st.warning(f"Couldn't fit lognormal distribution: {str(e)}")

#--------------------------------------------------------------------

st.set_page_config(page_title='LogNormalitY', page_icon=':tulip:')

cols = st.columns([0.6, 0.2, 0.2], gap='small', vertical_alignment="bottom")
with cols[0]:
    st.title('LogNormalitY')
with cols[2]:
    st.markdown('_aadm, 2025-05-13_')
st.divider()

dist_list = ['norm', 'triang', 'uniform']

cols = st.columns(3, gap='medium')
with cols[0]:
    dist_name = st.selectbox('distribution type', dist_list, index=0)
with cols[1]:
    ndist = st.number_input('number of datasets', value=3, min_value=2, max_value=30, step=1, format='%d')
with cols[2]:
    nsamples = st.number_input('number of samples', value=100, min_value=10, max_value=1000)

# Initialize session state variables if they don't exist
if 'figure' not in st.session_state:
    st.session_state.figure = None
    st.session_state.axes = None
    st.session_state.prod_samples = None
    st.session_state.recalculate = True

# Use a button to trigger recalculation
if st.button("Recalculate"):
    st.session_state.recalculate = True

nsamples = 1000
cm = mplcmap['brg']
color_indices = np.linspace(0, 1, ndist)
colors = [cm(i) for i in color_indices]
rng = np.random.default_rng()
mu = np.round(rng.uniform(low=5, high=20, size=ndist), 1)
sigma = np.round(abs(rng.normal(loc=1, scale=2, size=ndist)), 1)
symm = np.round(rng.uniform(low=0, high=1, size=ndist), 1)



if st.session_state.recalculate:
    figure, axes, prod_samples = cached_mult2dist(dist_name, mu, sigma, symm, nsamples)
    st.session_state.figure = figure
    st.session_state.axes = axes
    st.session_state.prod_samples = prod_samples
    st.session_state.recalculate = False
else:
    figure = st.session_state.figure
    axes = st.session_state.axes
    prod_samples = st.session_state.prod_samples

lognormal = st.checkbox('lognormal fit', value=True)

# Update the lognormal fit based on the checkbox
if st.session_state.axes is not None:
    update_lognormal_fit(st.session_state.axes[1], st.session_state.prod_samples, lognormal)

# Display the updated figure
if st.session_state.figure is not None:
    st.pyplot(fig=st.session_state.figure, use_container_width=True)
else:
    st.write("Please click 'Recalculate' to generate the plot.")

