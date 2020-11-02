from flask import Flask, render_template, url_for, request, redirect
from flask import jsonify
app = Flask(__name__, template_folder="templates")


@app.route('/', methods=["GET","POST"])
def index():

    if request.method == "POST":
        longitude = request.form["longitude"]
        latitude = request.form["latitude"]

        return lredirect(url_for("pathplanning", longitude=longitude, latitude=latitude))

    return render_template("index.html")

@app.route('/base')
def base():

    return render_template("base.html")


@app.route('/pathplanning', methods=['POST'])
def pathplanning():

    if request.method == "POST":
        start_waypoint = request.form['start_waypoint']
        end_waypoint = request.form['end_waypoint']
        polygon = request.form['polygon']

        ## REQUEST DATA TO REDEMET
        ## DO PATH PLANNING AND RETURN A ROUTE

        route = {'route', {
                        'type': 'geojson',
                        'data': {
                            'type': 'Feature',
                            'properties': {},
                            'geometry': {
                                'type': 'LineString',
                                'coordinates': [
                                    [-122.48369693756104, 37.83381888486939],
                                    [-122.48348236083984, 37.83317489144141],
                                    [-122.49378204345702, 37.83368330777276]
                                    ]
                                }}}}
        
        return jsonify(route)

    return render_template("main.html")


if __name__ == "__main__":
    app.run(debug=True) 