# limnotech_rating_curves.view.__init__

Viewing a rating: static figures and the interactive map.

    from limnotech_rating_curves.view import plot_site, build_map

    plot_site(rating)               # measurements, fitted curve, credible band
    build_map(sites)                # one HTML page, every site, click to compare

matplotlib and plotly are imported inside these functions, so no plotting backend is
loaded until a figure is requested.
