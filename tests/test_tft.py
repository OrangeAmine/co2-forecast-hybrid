"""Tests for TFT feature auto-classification."""

from src.models.tft import classify_tft_features


class TestClassifyTFTFeatures:
    """Tests for automatic feature classification into known/unknown reals."""

    def test_baseline_features(self):
        """Baseline 9 features should be classified correctly."""
        features = [
            "Noise", "Pression", "TemperatureExt", "Hrext",
            "Day_sin", "Day_cos", "Year_cos", "Year_sin", "dCO2",
        ]
        known, unknown = classify_tft_features(features, target="CO2")
        # sin/cos features should be known reals
        assert "Day_sin" in known
        assert "Day_cos" in known
        assert "Year_sin" in known
        assert "Year_cos" in known
        # Sensor readings should be unknown reals
        assert "Noise" in unknown
        assert "Pression" in unknown
        assert "TemperatureExt" in unknown
        assert "dCO2" in unknown
        # Target should be in unknown
        assert "CO2" in unknown

    def test_enhanced_features(self):
        """Exp3 enhanced features should classify weekday sin/cos as known."""
        features = [
            "Noise", "Pression", "TemperatureExt", "Hrext",
            "Day_sin", "Day_cos", "Year_cos", "Year_sin", "dCO2",
            "Weekday_sin", "Weekday_cos",
            "CO2_lag_12", "CO2_lag_72", "CO2_lag_288",
            "CO2_rolling_mean_12", "CO2_rolling_std_12",
            "CO2_rolling_mean_72", "CO2_rolling_std_72",
        ]
        known, unknown = classify_tft_features(features, target="CO2")
        # Weekday features should be known reals (they have _sin/_cos)
        assert "Weekday_sin" in known
        assert "Weekday_cos" in known
        # Lag and rolling features should be unknown reals
        assert "CO2_lag_12" in unknown
        assert "CO2_rolling_mean_12" in unknown
        # Total counts
        assert len(known) == 6  # Day_sin, Day_cos, Year_sin, Year_cos, Weekday_sin, Weekday_cos
        assert "CO2" in unknown  # Target auto-added

    def test_target_auto_added(self):
        """Target should be added to unknown reals if not in feature list."""
        features = ["Noise", "Day_sin"]
        known, unknown = classify_tft_features(features, target="CO2")
        assert "CO2" in unknown

    def test_target_not_duplicated(self):
        """If target is already in features, it should not be duplicated."""
        features = ["CO2", "Noise", "Day_sin"]
        known, unknown = classify_tft_features(features, target="CO2")
        assert unknown.count("CO2") == 1
