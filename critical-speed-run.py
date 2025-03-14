import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

# Set page configuration
st.set_page_config(
    page_title="Critical Speed Calculator",
    page_icon="🏃‍♂️",
    layout="wide"
)

# Apply custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

.main {
    background-color: #FFFFFF;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    color: #E6754E;
}

.stButton>button {
    background-color: #E6754E;
    color: white;
    font-family: 'Montserrat', sans-serif;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
}

.stButton>button:hover {
    background-color: #c45d3a;
}

.highlight {
    color: #E6754E;
    font-weight: 600;
}

.result-box {
    background-color: #f8f8f8;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #E6754E;
}

footer {
    font-family: 'Montserrat', sans-serif;
    font-size: 12px;
    color: #888888;
    text-align: center;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("Critical Speed Calculator for Runners")
st.markdown('<p class="highlight">Science-based running performance metrics</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=Your+Logo", width=150)
    st.markdown("## Test Protocols")
    test_method = st.selectbox(
        "Select Testing Method",
        ["3-Minute All Out Test", "Time Trials Method", "3/5-Minute Test", "Ramp Test", "Time to Exhaustion Test"]
    )
    
    st.markdown("## Runner Information")
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=70.0, step=0.1)
    
    experience_level = st.radio(
        "Experience Level",
        ["Beginner", "Intermediate", "Advanced", "Elite"]
    )
    
    with st.expander("Advanced Settings"):
        show_formulas = st.checkbox("Show calculation formulas", value=False)
        show_references = st.checkbox("Show scientific references", value=True)
        distance_unit = st.radio("Distance Unit", ["Meters", "Kilometers", "Miles"])

# Helper functions for unit conversion
def convert_to_meters(distance, unit):
    if unit == "Kilometers":
        return distance * 1000
    elif unit == "Miles":
        return distance * 1609.34
    return distance

def convert_from_meters(distance, unit):
    if unit == "Kilometers":
        return distance / 1000
    elif unit == "Miles":
        return distance / 1609.34
    return distance

def format_pace(speed_mps, unit):
    """Convert speed in m/s to pace format (min:sec per km or mile)"""
    if unit == "Kilometers":
        pace_seconds = 1000 / speed_mps
    elif unit == "Miles":
        pace_seconds = 1609.34 / speed_mps
    else:  # Meters doesn't make sense for pace
        pace_seconds = 1000 / speed_mps
    
    minutes = int(pace_seconds // 60)
    seconds = int(pace_seconds % 60)
    return f"{minutes}:{seconds:02d}"

# Helper functions for critical speed calculations
def calculate_cs_from_3min(max_speed, end_speed):
    """Calculate critical speed from 3-minute all out test"""
    cs = end_speed
    d_prime = (max_speed - end_speed) * 180  # 3 minutes in seconds
    return cs, d_prime

def calculate_cs_from_time_trials(distances, times):
    """Calculate critical speed from time-trial method"""
    # Linear regression of distance vs time model
    def distance_model(t, cs, d_prime):
        return cs * t + d_prime
    
    popt, _ = curve_fit(distance_model, times, distances)
    cs, d_prime = popt
    return cs, d_prime

def calculate_cs_from_3_5min(distance_3min, distance_5min):
    """Calculate critical speed from 3/5-minute test"""
    time_diff = 120  # 5min - 3min = 2min = 120sec
    cs = (distance_5min - distance_3min) / time_diff
    d_prime = distance_3min - (cs * 180)  # 3min = 180sec
    return cs, d_prime

def calculate_cs_from_ramp(final_speed, time_to_exhaustion, ramp_rate):
    """Calculate critical speed from ramp test"""
    # Based on analogous calculations from critical power literature
    cs = final_speed - (0.5 * ramp_rate * time_to_exhaustion / 60)
    d_prime = 0.5 * ramp_rate * (time_to_exhaustion / 60) ** 2
    return cs, d_prime

def calculate_cs_from_tte(speeds, times):
    """Calculate critical speed from time to exhaustion tests"""
    if len(speeds) < 2 or len(times) < 2:
        return 0, 0
        
    def speed_time_model(t, cs, d_prime):
        return cs + d_prime / t
    
    popt, _ = curve_fit(speed_time_model, times, speeds)
    cs, d_prime = popt
    return cs, d_prime

def calculate_training_paces(cs, unit):
    """Calculate training paces based on critical speed"""
    # Pace zones based on scientific literature for running
    paces = {
        "Recovery": (0.60 * cs, 0.70 * cs),
        "Easy/Aerobic": (0.70 * cs, 0.80 * cs),
        "Moderate": (0.80 * cs, 0.87 * cs),
        "Threshold": (0.87 * cs, 0.93 * cs),
        "Critical Speed": (0.93 * cs, 1.00 * cs),
        "Interval": (1.00 * cs, 1.10 * cs),
        "Repetition": (1.10 * cs, 1.20 * cs)
    }
    
    formatted_paces = {}
    for zone, (lower, upper) in paces.items():
        lower_pace = format_pace(lower, unit)
        upper_pace = format_pace(upper, unit)
        
        if unit == "Kilometers":
            formatted_paces[zone] = f"{lower_pace} - {upper_pace} min/km"
        elif unit == "Miles":
            formatted_paces[zone] = f"{lower_pace} - {upper_pace} min/mile"
        else:
            formatted_paces[zone] = f"{lower_pace} - {upper_pace} min/km"
            
    return formatted_paces

def predict_race_times(cs, d_prime, unit):
    """Predict race times for common distances based on CS and D'"""
    distances = {
        "800m": 800,
        "1500m": 1500,
        "Mile": 1609.34,
        "3000m": 3000,
        "5K": 5000,
        "10K": 10000,
        "Half Marathon": 21097.5,
        "Marathon": 42195
    }
    
    predictions = {}
    for race, dist in distances.items():
        # Using the inverse of the CS-D' relationship: t = d/CS - D'/CS²
        time_seconds = dist / cs - d_prime / (cs * cs)
        
        if time_seconds > 0:
            # Calculate average pace for this distance
            avg_speed = dist / time_seconds
            
            if unit == "Miles" and (race == "5K" or race == "10K" or race == "Half Marathon" or race == "Marathon"):
                # Convert to pace per mile for common longer races when using miles
                race_distance_miles = dist / 1609.34
                pace_per_mile = format_pace(avg_speed, "Miles")
                pace_info = f" (pace: {pace_per_mile}/mi)"
            else:
                # Default to pace per km
                pace_per_km = format_pace(avg_speed, "Kilometers")
                pace_info = f" (pace: {pace_per_km}/km)"
            
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = int(time_seconds % 60)
            
            if hours > 0:
                predictions[race] = f"{hours}:{minutes:02d}:{seconds:02d}{pace_info}"
            else:
                predictions[race] = f"{minutes}:{seconds:02d}{pace_info}"
        else:
            predictions[race] = "N/A"
            
    return predictions

def d_prime_balance(speed_data, cs, d_prime, tau=300):
    """
    Calculate D' balance over a run
    Based on similar methodology to W' balance in cycling
    
    Parameters:
    speed_data: array of speed data in m/s
    cs: critical speed in m/s
    d_prime: D' (anaerobic distance capacity) in meters
    tau: time constant for D' reconstitution
    
    Returns:
    Array of D' balance values
    """
    d_prime_balance = np.ones(len(speed_data)) * d_prime
    for i in range(1, len(speed_data)):
        expenditure = max(0, (speed_data[i-1] - cs)) # D' expenditure
        recovery = min(0, (speed_data[i-1] - cs))    # D' recovery
        
        # Apply differential equation similar to Skiba's W' model
        if expenditure > 0:
            d_prime_balance[i] = d_prime_balance[i-1] - expenditure
        else:
            # D' reconstitution
            d_prime_balance[i] = d_prime - (d_prime - d_prime_balance[i-1]) * np.exp(recovery / (tau * cs))
            
        # Ensure D' balance doesn't exceed D'
        d_prime_balance[i] = min(d_prime, max(0, d_prime_balance[i]))
        
    return d_prime_balance

# Main content based on selected method
if test_method == "3-Minute All Out Test":
    st.markdown("## 3-Minute All-Out Test")
    
    st.markdown("""
    The 3-minute all-out test is a simplified protocol to determine Critical Speed (CS) and D′ (anaerobic distance capacity).
    
    ### Protocol:
    1. Perform a thorough warm-up
    2. Start the test with maximal effort from the beginning
    3. Continue with maximum possible effort for the entire 3 minutes
    4. Record your maximum speed and average speed for the last 30 seconds
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_speed = st.number_input("Maximum Speed (m/s)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
        end_speed = st.number_input("Average Speed in Last 30s (m/s)", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
        
        # Add unit conversion UI if needed
        st.markdown("### Speed Entry Options")
        speed_entry = st.radio("Speed Entry Method", ["Direct (m/s)", "Pace"])
        
        if speed_entry == "Pace":
            pace_unit = "min/km" if distance_unit != "Miles" else "min/mile"
            max_pace_min = st.number_input("Maximum Pace Minutes", min_value=1, max_value=10, value=3)
            max_pace_sec = st.number_input("Maximum Pace Seconds", min_value=0, max_value=59, value=0)
            end_pace_min = st.number_input("End Pace Minutes", min_value=1, max_value=10, value=4)
            end_pace_sec = st.number_input("End Pace Seconds", min_value=0, max_value=59, value=0)
            
            # Convert pace to speed
            max_pace_seconds = max_pace_min * 60 + max_pace_sec
            end_pace_seconds = end_pace_min * 60 + end_pace_sec
            
            if distance_unit == "Miles":
                max_speed = 1609.34 / max_pace_seconds
                end_speed = 1609.34 / end_pace_seconds
            else:
                max_speed = 1000 / max_pace_seconds
                end_speed = 1000 / end_pace_seconds
        
        if st.button("Calculate"):
            cs, d_prime = calculate_cs_from_3min(max_speed, end_speed)
            
            with col2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"**Critical Speed (CS):** {cs:.2f} m/s")
                
                # Convert to pace
                cs_pace_km = format_pace(cs, "Kilometers")
                cs_pace_mile = format_pace(cs, "Miles")
                
                st.markdown(f"**Critical Pace:** {cs_pace_km} min/km ({cs_pace_mile} min/mile)")
                st.markdown(f"**D′ (Anaerobic Distance Capacity):** {d_prime:.0f} meters")
                
                # Calculate and display training paces
                st.markdown("### Training Paces")
                paces = calculate_training_paces(cs, distance_unit)
                for zone, pace_range in paces.items():
                    st.markdown(f"**{zone}:** {pace_range}")
                
                # Predict race times
                st.markdown("### Predicted Race Times")
                predictions = predict_race_times(cs, d_prime, distance_unit)
                for race, time in predictions.items():
                    st.markdown(f"**{race}:** {time}")
                    
                st.markdown('</div>', unsafe_allow_html=True)
    
    if show_formulas:
        st.markdown("### Calculation Method")
        st.markdown("""
        - Critical Speed (CS) = Average speed during the final 30 seconds
        - D′ = (Maximum speed - CS) × 180 seconds
        """)
    
    if show_references:
        st.markdown("### Scientific References")
        st.markdown("""
        1. Pettitt, R. W., Jamnick, N., & Clark, I. E. (2012). 3-min all-out exercise test for running. International Journal of Sports Medicine, 33(6), 426-431.
        
        2. Broxterman, R. M., Ade, C. J., Poole, D. C., Harms, C. A., & Barstow, T. J. (2013). A single test for the determination of parameters of the speed-time relationship for running. Respiratory Physiology & Neurobiology, 185(2), 380-385.
        """)

elif test_method == "Time Trials Method":
    st.markdown("## Time Trials Method")
    
    st.markdown("""
    This method requires performing time trials at different distances to determine Critical Speed and D′.
    
    ### Protocol:
    1. Perform 3-5 maximal effort runs of different distances
    2. Record your time for each distance
    3. Rest at least 24 hours between trials
    4. Input the data below to calculate CS and D′
    """)
    
    num_trials = st.number_input("Number of Time Trials", min_value=2, max_value=5, value=3)
    
    cols = st.columns(num_trials)
    distances = []
    times = []
    
    for i in range(num_trials):
        with cols[i]:
            st.markdown(f"### Trial {i+1}")
            
            # Distance entry with unit conversion
            distance_val = st.number_input(f"Distance ({distance_unit})", 
                                         min_value=0.1, 
                                         max_value=50.0 if distance_unit != "Meters" else 10000.0, 
                                         value=1.0 if distance_unit != "Meters" else 400.0,
                                         step=0.1,
                                         key=f"dist_{i}")
            
            distance_meters = convert_to_meters(distance_val, distance_unit)
            
            # Time entry
            minutes = st.number_input(f"Minutes", min_value=0, max_value=180, value=3 if i == 0 else 5 if i == 1 else 10, key=f"min_{i}")
            seconds = st.number_input(f"Seconds", min_value=0, max_value=59, value=0, key=f"sec_{i}")
            
            time_seconds = minutes * 60 + seconds
            
            distances.append(distance_meters)
            times.append(time_seconds)
    
    if st.button("Calculate"):
        if num_trials >= 2:
            cs, d_prime = calculate_cs_from_time_trials(distances, times)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"**Critical Speed (CS):** {cs:.2f} m/s")
                
                # Convert to pace
                cs_pace_km = format_pace(cs, "Kilometers")
                cs_pace_mile = format_pace(cs, "Miles")
                
                st.markdown(f"**Critical Pace:** {cs_pace_km} min/km ({cs_pace_mile} min/mile)")
                st.markdown(f"**D′ (Anaerobic Distance Capacity):** {d_prime:.0f} meters")
                
                # Calculate and display training paces
                st.markdown("### Training Paces")
                paces = calculate_training_paces(cs, distance_unit)
                for zone, pace_range in paces.items():
                    st.markdown(f"**{zone}:** {pace_range}")
                
                # Predict race times
                st.markdown("### Predicted Race Times")
                predictions = predict_race_times(cs, d_prime, distance_unit)
                for race, time in predictions.items():
                    st.markdown(f"**{race}:** {time}")
                    
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Plot distance-time relationship
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot actual data points
                ax.scatter(times, distances, color='#E6754E', s=100, label='Time Trials')
                
                # Generate points for the linear relationship
                x_curve = np.linspace(0, max(times) * 1.2, 100)
                y_curve = cs * x_curve + d_prime
                
                # Plot the fitted line
                ax.plot(x_curve, y_curve, 'b-', label='Distance-Time Line')
                
                # Set axis labels based on user's unit preference
                ax.set_xlabel('Time (seconds)')
                
                if distance_unit == "Kilometers":
                    ax.set_ylabel('Distance (km)')
                    # Convert y-axis from meters to kilometers
                    ax.set_yticks(np.arange(0, max(distances) * 1.2, 1000))
                    ax.set_yticklabels([f"{x/1000:.0f}" for x in np.arange(0, max(distances) * 1.2, 1000)])
                elif distance_unit == "Miles":
                    ax.set_ylabel('Distance (miles)')
                    # Convert y-axis from meters to miles
                    ax.set_yticks(np.arange(0, max(distances) * 1.2, 1609.34))
                    ax.set_yticklabels([f"{x/1609.34:.0f}" for x in np.arange(0, max(distances) * 1.2, 1609.34)])
                else:
                    ax.set_ylabel('Distance (meters)')
                
                ax.set_title('Distance-Time Relationship')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
                # Also show in tabular format
                st.markdown("### Trial Data")
                trial_data = []
                for i in range(num_trials):
                    trial_data.append({
                        "Distance": f"{convert_from_meters(distances[i], distance_unit):.2f} {distance_unit}",
                        "Time": f"{times[i] // 60}:{times[i] % 60:02d}",
                        "Speed": f"{distances[i] / times[i]:.2f} m/s"
                    })
                
                st.table(pd.DataFrame(trial_data))
        else:
            st.error("At least 2 time trials are required for calculation")
    
    if show_formulas:
        st.markdown("### Calculation Method")
        st.markdown("""
        The linear distance-time model is used:
        
        d = CS × t + D′
        
        Where:
        - d is the distance covered
        - t is the time in seconds
        - CS is critical speed
        - D′ is the anaerobic distance capacity
        
        The parameters are determined through linear regression.
        """)
    
    if show_references:
        st.markdown("### Scientific References")
        st.markdown("""
        1. Jones, A. M., & Vanhatalo, A. (2017). The 'Critical Power' concept: Applications to sports performance with a focus on intermittent high-intensity exercise. Sports Medicine, 47(Suppl 1), 65-78.
        
        2. Florence, S., & Weir, J. P. (1997). Relationship of critical velocity to marathon running performance. European Journal of Applied Physiology and Occupational Physiology, 75(3), 274-278.
        
        3. Fukuba, Y., & Whipp, B. J. (1999). A metabolic limit on the ability to make up for lost time in endurance events. Journal of Applied Physiology, 87(2), 853-861.
        """)

elif test_method == "3/5-Minute Test":
    st.markdown("## 3/5-Minute Test")
    
    st.markdown("""
    This method uses two time-based all-out efforts to determine Critical Speed and D′.
    
    ### Protocol:
    1. Perform a 3-minute all-out run, measuring the total distance covered
    2. After sufficient recovery (24-48 hours), perform a 5-minute all-out run
    3. Input both distances to calculate CS and D′
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distance entry with unit conversion
        distance_3min_val = st.number_input(f"3-Minute Distance ({distance_unit})", 
                                     min_value=0.1, 
                                     max_value=50.0 if distance_unit != "Meters" else 5000.0, 
                                     value=0.8 if distance_unit != "Meters" else 800.0,
                                     step=0.1)
        
        distance_5min_val = st.number_input(f"5-Minute Distance ({distance_unit})", 
                                     min_value=0.1, 
                                     max_value=50.0 if distance_unit != "Meters" else 5000.0, 
                                     value=1.3 if distance_unit != "Meters" else 1300.0,
                                     step=0.1)
        
        # Convert to meters
        distance_3min = convert_to_meters(distance_3min_val, distance_unit)
        distance_5min = convert_to_meters(distance_5min_val, distance_unit)
        
        if st.button("Calculate"):
            cs, d_prime = calculate_cs_from_3_5min(distance_3min, distance_5min)
            
            with col2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"**Critical Speed (CS):** {cs:.2f} m/s")
                
                # Convert to pace
                cs_pace_km = format_pace(cs, "Kilometers")
                cs_pace_mile = format_pace(cs, "Miles")
                
                st.markdown(f"**Critical Pace:** {cs_pace_km} min/km ({cs_pace_mile} min/mile)")
                st.markdown(f"**D′ (Anaerobic Distance Capacity):** {d_prime:.0f} meters")
                
                # Calculate and display training paces
                st.markdown("### Training Paces")
                paces = calculate_training_paces(cs, distance_unit)
                for zone, pace_range in paces.items():
                    st.markdown(f"**{zone}:** {pace_range}")
                
                # Predict race times
                st.markdown("### Predicted Race Times")
                predictions = predict_race_times(cs, d_prime, distance_unit)
                for race, time in predictions.items():
                    st.markdown(f"**{race}:** {time}")
                    
                st.markdown('</div>', unsafe_allow_html=True)
    
    if show_formulas:
        st.markdown("### Calculation Method")
        st.markdown("""
        Using the two distances covered:
        
        - CS = (5-min distance - 3-min distance) / 120 seconds
        - D′ = 3-min distance - (CS × 180 seconds)
        
        This is based on the linear distance-time relationship, using two data points.
        """)
    
    if show_references:
        st.markdown("### Scientific References")
        st.markdown("""
        1. Burnley, M., Doust, J. H., & Vanhatalo, A. (2006). A 3-min all-out test to determine peak oxygen uptake and the maximal steady state. Medicine and Science in Sports and Exercise, 38(11), 1995-2003.
        
        2. Jones, A. M., & Vanhatalo, A. (2017). The 'Critical Power' concept: Applications to sports performance with a focus on intermittent high-intensity exercise. Sports Medicine, 47(Suppl 1), 65-78.
        """)

elif test_method == "Ramp Test":
    st.markdown("## Ramp Test")
    
    st.markdown("""
    The ramp test is an incremental exercise test where speed is progressively increased until exhaustion.
    
    ### Protocol:
    1. Start at a low running speed
    2. Increase speed at a constant rate (e.g., 0.5 km/h per minute)
    3. Continue until exhaustion
    4. Record final speed and time to exhaustion
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        starting_speed_kmh = st.number_input("Starting Speed (km/h)", min_value=4.0, max_value=12.0, value=8.0)
        ramp_rate_kmh = st.number_input("Ramp Rate (km/h per minute)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        time_to_exhaustion = st.number_input("Time to Exhaustion (seconds)", min_value=300, max_value=1800, value=720)
        
        # Convert to m/s for calculations
        starting_speed = starting_speed_kmh / 3.6
        ramp_rate = ramp_rate_kmh / 3.6
        
        if st.button("Calculate"):
            final_speed = starting_speed + (ramp_rate * time_to_exhaustion / 60)
            cs, d_prime = calculate_cs_from_ramp(final_speed, time_to_exhaustion, ramp_rate)
            
            with col2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"**Final Speed:** {final_speed*3.6:.1f} km/h ({final_speed:.2f} m/s)")
                st.markdown(f"**Critical Speed (CS):** {cs:.2f} m/s ({cs*3.6:.1f} km/h)")
                
                # Convert to pace
                cs_pace_km = format_pace(cs, "Kilometers")
                cs_pace_mile = format_pace(cs, "Miles")
                
                st.markdown(f"**Critical Pace:** {cs_pace_km} min/km ({cs_pace_mile} min/mile)")
                st.markdown(f"**D′ (Anaerobic Distance Capacity):** {d_prime:.0f} meters")
                
                # Calculate and display training paces
                st.markdown("### Training Paces")
                paces = calculate_training_paces(cs, distance_unit)
                for zone, pace_range in paces.items():
                    st.markdown(f"**{zone}:** {pace_range}")
                
                # Predict race times
                st.markdown("### Predicted Race Times")
                predictions = predict_race_times(cs, d_prime, distance_unit)
                for race, time in predictions.items():
                    st.markdown(f"**{race}:** {time}")
                    
                st.markdown('</div>', unsafe_allow_html=True)
    
    if show_formulas:
        st.markdown("### Calculation Method")
        st.markdown("""
        For a ramp test with constant increment rate:
        
        - Critical Speed (CS) = Smax - (0.5 × ramp rate × TTE)
        - D′ = 0.5 × ramp rate × TTE²
        
        Where:
        - Smax is the final speed at exhaustion
        - TTE is time to exhaustion in minutes
        - ramp rate is in m/s per minute
        """)
    
    if show_references:
        st.markdown("### Scientific References")
        st.markdown("""
        1. Hughson, R. L., Orok, C. J., & Staudt, L. E. (1984). A high velocity treadmill running test to assess endurance running potential. International Journal of Sports Medicine, 5(1), 23-25.
        
        2. Poole, D. C., Burnley, M., Vanhatalo, A., Rossiter, H. B., & Jones, A. M. (2016). Critical power: An important fatigue threshold in exercise physiology. Medicine and Science in Sports and Exercise, 48(11), 2320-2334.
        """)

elif test_method == "Time to Exhaustion Test":
    st.markdown("## Time to Exhaustion Test")
    
    st.markdown("""
    This method requires performing multiple constant-speed runs until exhaustion at different intensities.
    
    ### Protocol:
    1. Perform runs at different constant speeds
    2. Record time to exhaustion at each speed
    3. Allow sufficient recovery between tests (24-48 hours)
    4. Input the data below to calculate CS and D′
    """)
    
    num_tests = st.number_input("Number of Tests", min_value=2, max_value=5, value=3)
    
    cols = st.columns(num_tests)
    speeds = []
    tte_values = []
    
    for i in range(num_tests):
        with cols[i]:
            st.markdown(f"### Test {i+1}")
            
            # Speed entry
            speed_kmh = st.number_input(f"Speed (km/h)", min_value=5.0, max_value=25.0, value=14.0 - i*2.0, key=f"speed_{i}")
            speed_ms = speed_kmh / 3.6  # Convert to m/s
            
            # Time entry
            minutes = st.number_input(f"Minutes to Exhaustion", min_value=0, max_value=60, value=5 + i*5, key=f"tte_min_{i}")
            seconds = st.number_input(f"Seconds to Exhaustion", min_value=0, max_value=59, value=0, key=f"tte_sec_{i}")
            
            time_seconds = minutes * 60 + seconds
            
            speeds.append(speed_ms)
            tte_values.append(time_seconds)
    
    if st.button("Calculate"):
        if num_tests >= 2:
            cs, d_prime = calculate_cs_from_tte(speeds, tte_values)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f"### Results")
                st.markdown(f"**Critical Speed (CS):** {cs:.2f} m/s ({cs*3.6:.1f} km/h)")
                
                # Convert to pace
                cs_pace_km = format_pace(cs, "Kilometers")
                cs_pace_mile = format_pace(cs, "Miles")
                
                st.markdown(f"**Critical Pace:** {cs_pace_km} min/km ({cs_pace_mile} min/mile)")
                st.markdown(f"**D′ (Anaerobic Distance Capacity):** {d_prime:.0f} meters")
                
                # Calculate and display training paces
                st.markdown("### Training Paces")
                paces = calculate_training_paces(cs, distance_unit)
                for zone, pace_range in paces.items():
                    st.markdown(f"**{zone}:** {pace_range}")
                
                # Predict race times
                st.markdown("### Predicted Race Times")
                predictions = predict_race_times(cs, d_prime, distance_unit)
                for race, time in predictions.items():
                    st.markdown(f"**{race}:** {time}")
                    
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Plot speed-time relationship
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot actual data points
                ax.scatter(tte_values, speeds, color='#E6754E', s=100, label='TTE Tests')
                
                # Generate points for the hyperbolic curve
                x_curve = np.linspace(min(tte_values) * 0.8, max(tte_values) * 1.2, 100)
                y_curve = [cs + (d_prime / x) for x in x_curve]
                
                # Plot the fitted curve
                ax.plot(x_curve, y_curve, 'b-', label='Speed-Time Curve')
                
                # Add horizontal line for CS
                ax.axhline(y=cs, color='g', linestyle='--', label=f'CS: {cs:.2f} m/s')
                
                # Convert y-axis from m/s to km/h for readability
                ax.set_yticks(np.arange(0, max(speeds) * 1.2, 0.5))
                ax.set_yticklabels([f"{x*3.6:.1f}" for x in np.arange(0, max(speeds) * 1.2, 0.5)])
                
                ax.set_xlabel('Time to Exhaustion (seconds)')
                ax.set_ylabel('Speed (km/h)')
                ax.set_title('Speed-Time Relationship')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
        else:
            st.error("At least 2 tests are required for calculation")
    
    if show_formulas:
        st.markdown("### Calculation Method")
        st.markdown("""
        The hyperbolic relationship between speed and time to exhaustion is used:
        
        S = CS + D′/t
        
        Where:
        - S is the constant speed
        - t is the time to exhaustion in seconds
        - CS is critical speed
        - D′ is the anaerobic distance capacity
        
        The parameters are determined through hyperbolic curve fitting.
        """)
    
    if show_references:
        st.markdown("### Scientific References")
        st.markdown("""
        1. Hill, D. W. (1993). The critical power concept. Sports Medicine, 16(4), 237-254.
        
        2. Smith, C. G., & Jones, A. M. (2001). The relationship between critical velocity, maximal lactate steady-state velocity and lactate turnpoint velocity in runners. European Journal of Applied Physiology, 85(1-2), 19-26.
        
        3. Poole, D. C., Ward, S. A., Gardner, G. W., & Whipp, B. J. (1988). Metabolic and respiratory profile of the upper limit for prolonged exercise in man. Ergonomics, 31(9), 1265-1279.
        """)

# D' Balance Simulator
st.markdown("---")
st.markdown("## D′ Balance Simulator")
st.markdown("""
This simulator helps you understand how D′ (your anaerobic distance capacity) gets depleted and recharged during a run based on your estimated Critical Speed.
""")

if 'cs' in locals() and 'd_prime' in locals():
    simulate_enabled = True
    default_cs = cs
    default_dprime = d_prime
else:
    simulate_enabled = False
    default_cs = 4.0  # m/s
    default_dprime = 250  # meters

col1, col2 = st.columns([1, 1])

with col1:
    if not simulate_enabled:
        st.markdown("#### Enter your values or calculate them using the methods above")
        sim_cs = st.number_input("Critical Speed (m/s)", min_value=2.0, max_value=7.0, value=default_cs, step=0.1)
        sim_dprime = st.number_input("D′ (m)", min_value=50, max_value=500, value=default_dprime)
    else:
        sim_cs = default_cs
        sim_dprime = default_dprime
        st.markdown(f"#### Using calculated values: CS = {sim_cs:.2f} m/s, D′ = {sim_dprime:.0f}m")
    
    st.markdown("#### Simulation Parameters")
    duration = st.slider("Run Duration (minutes)", min_value=5, max_value=60, value=20)
    
    # Create a simple run profile
    profile_type = st.selectbox("Run Profile", ["Steady Pace", "Intervals", "Variable Pace", "Race Simulation"])
    
    # Define all potential parameters upfront to avoid state issues
    intensity = 95
    work_intensity = 120
    rest_intensity = 70
    interval_length = 120
    rest_length = 60
    base_intensity = 85
    variability = 15
    
    # Show relevant parameters based on profile
    if profile_type == "Steady Pace":
        intensity = st.slider("Intensity (% of CS)", min_value=70, max_value=110, value=95)
    elif profile_type == "Intervals":
        work_intensity = st.slider("Work Interval Intensity (% of CS)", min_value=100, max_value=150, value=120)
        rest_intensity = st.slider("Recovery Interval Intensity (% of CS)", min_value=50, max_value=90, value=70)
        interval_length = st.slider("Interval Length (seconds)", min_value=30, max_value=300, value=120)
        rest_length = st.slider("Rest Length (seconds)", min_value=30, max_value=300, value=60)
    elif profile_type == "Variable Pace":
        base_intensity = st.slider("Base Intensity (% of CS)", min_value=70, max_value=100, value=85)
        variability = st.slider("Variability (%)", min_value=5, max_value=30, value=15)
    # Race simulation doesn't need additional parameters
    
    if st.button("Run Simulation"):
        # Generate speed data based on profile type
        time_axis = np.linspace(0, duration*60, duration*60)  # second-by-second
        
        if profile_type == "Steady Pace":
            # Using pre-defined intensity from slider
            speed_data = np.ones_like(time_axis) * sim_cs * (intensity/100)
            
        elif profile_type == "Intervals":
            # Using pre-defined work_intensity, rest_intensity, interval_length, rest_length from sliders
            speed_data = np.zeros_like(time_axis)
            for i in range(len(time_axis)):
                cycle_position = i % (interval_length + rest_length)
                if cycle_position < interval_length:
                    speed_data[i] = sim_cs * (work_intensity/100)
                else:
                    speed_data[i] = sim_cs * (rest_intensity/100)
        
        elif profile_type == "Variable Pace":
            # Using pre-defined base_intensity and variability from sliders
            # Create a variable run with random fluctuations
            from scipy.ndimage import gaussian_filter1d
            random_variations = np.random.normal(0, variability/100, size=len(time_axis))
            smoothed_variations = gaussian_filter1d(random_variations, sigma=30)  # Smoothing factor
            
            speed_data = ((base_intensity/100) + smoothed_variations) * sim_cs
            speed_data = np.clip(speed_data, 0.5*sim_cs, 1.5*sim_cs)  # Limit the range
            
        elif profile_type == "Race Simulation":
            # Simulates a race with a steady start, some surges, and a finishing kick
            speed_data = np.ones_like(time_axis) * 0.9 * sim_cs  # Base at 90% CS
            
            # Add some surges (3-4 random surges)
            num_surges = np.random.randint(3, 5)
            for _ in range(num_surges):
                surge_start = np.random.randint(5*60, (duration-3)*60)
                surge_duration = np.random.randint(30, 120)  # 30s to 2min surges
                surge_intensity = np.random.uniform(1.1, 1.3)  # 110-130% CS
                speed_data[surge_start:surge_start+surge_duration] = sim_cs * surge_intensity
                
            # Add final kick in last 30-60 seconds
            kick_start = (duration * 60) - np.random.randint(30, 60)
            kick_duration = np.random.randint(20, 40)
            speed_data[kick_start:kick_start+kick_duration] = sim_cs * 1.3  # 130% of CS for final kick
        
        # Calculate D' balance
        d_balance = d_prime_balance(speed_data, sim_cs, sim_dprime)
        
        # Plot results
        with col2:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            color = '#E6754E'
            ax1.set_xlabel('Time (minutes)')
            ax1.set_ylabel('Speed (km/h)', color=color)
            ax1.plot(time_axis/60, speed_data*3.6, color=color, alpha=0.7)  # Convert m/s to km/h
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.axhline(y=sim_cs*3.6, color=color, linestyle='--', alpha=0.7, label=f'CS: {sim_cs*3.6:.1f} km/h')
            
            # Create second y-axis for D' balance
            ax2 = ax1.twinx()
            color = 'blue'
            ax2.set_ylabel('D′ Balance (m)', color=color)
            ax2.plot(time_axis/60, d_balance, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.axhline(y=sim_dprime, color=color, linestyle='--', alpha=0.7, label=f'D′: {sim_dprime:.0f}m')
            
            # Add a D' depletion danger zone
            ax2.axhspan(0, 20, color='red', alpha=0.2, label='Danger Zone')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.title('Running Speed and D′ Balance Simulation')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Summary statistics
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown("### Simulation Summary")
            avg_speed_kmh = np.mean(speed_data) * 3.6
            avg_pace_km = format_pace(np.mean(speed_data), "Kilometers")
            
            st.markdown(f"**Average Speed:** {avg_speed_kmh:.1f} km/h ({avg_pace_km} min/km)")
            st.markdown(f"**Average Intensity:** {np.mean(speed_data)/sim_cs*100:.1f}% of CS")
            st.markdown(f"**Time Above CS:** {np.sum(speed_data > sim_cs)/60:.1f} minutes")
            st.markdown(f"**Minimum D′ Balance:** {min(d_balance):.0f} meters")
            st.markdown(f"**D′ Expended:** {sim_dprime - min(d_balance):.0f} meters ({(sim_dprime - min(d_balance))/sim_dprime*100:.1f}% of total)")
            
            # Estimated fatigue level
            fatigue_level = (sim_dprime - min(d_balance)) / sim_dprime
            if fatigue_level < 0.5:
                fatigue_status = "Low fatigue - You could maintain this effort or increase intensity"
            elif fatigue_level < 0.8:
                fatigue_status = "Moderate fatigue - Sustainable but challenging"
            else:
                fatigue_status = "High fatigue - Near exhaustion, significantly reduced performance"
            
            st.markdown(f"**Fatigue Status:** {fatigue_status}")
            st.markdown('</div>', unsafe_allow_html=True)

# Add explanation of results
st.markdown("---")
st.markdown("## Interpreting Your Results")

st.markdown("""
### Critical Speed (CS)
Critical Speed represents the highest intensity you can sustain for a very long time (theoretically 30-60 minutes) without continual fatigue accumulation. It's closely related to your lactate threshold and is a fundamental metric for endurance performance.

### D′ (D-prime)
D′ represents your finite anaerobic distance capacity - the amount of distance you can cover above your Critical Speed before exhaustion. Think of it as your "battery" of high-intensity energy.

### How to use these metrics:
1. **Pacing:** In longer races, stay at or slightly below your CS to avoid depleting D′
2. **Intervals:** Design interval sessions that target CS (to improve it) or D′ (to expand it)
3. **Race strategy:** For distance events, manage your D′ expenditure carefully - save some for hills and finishing sprint
4. **Training zones:** Use CS to set precise, physiologically meaningful training zones

### What makes a good CS and D′?
| Level | CS (m/s) | CS (min/km) | D′/kg (m/kg) |
|-------|----------|-------------|--------------|
| Recreational | 2.5-3.5 | 4:45-6:40 | 2.0-3.0 |
| Competitive | 3.5-4.5 | 3:42-4:45 | 2.5-4.0 |
| Elite | 4.5-5.5 | 3:01-3:42 | 3.0-5.0 |
| World Class | 5.5+ | <3:01 | 4.0-6.0 |

*Note: These values vary based on gender, age, and specialization.*
""")

# Add a footer
st.markdown("---")
st.markdown('<footer>Critical Speed Calculator for Runners © 2025</footer>', unsafe_allow_html=True)
