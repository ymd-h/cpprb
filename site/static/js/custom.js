(function(){
    const path = location.pathname

    var gl = $.get("https://gitlab.com/api/v4/projects/ymd_h%2Fcpprb");

    $(document).ready(function(){
	if(gl){
	    gl.then((data)=>{
		$("section#shortcuts a.padding > i.fa-gitlab").parent()
		    .append(" (<i class='fas fa-star fa-fw'></i>"
			    + data["star_count"]
			    + "<i class='fas fa-code-branch fa-fw'></i>"
			    + data["forks_count"]
			    + ")");
	    });
	}
    });
})();
