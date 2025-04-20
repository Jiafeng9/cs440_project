/**
 * Main JavaScript file for RoboLab platform
 * Handles common functionality across all pages
 */

document.addEventListener('DOMContentLoaded', function() {

    
    // Active link highlighting based on current URL
    const currentLocation = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (linkPath === currentLocation || 
            (currentLocation === '/' && linkPath === '/') ||
            (linkPath !== '/' && currentLocation.startsWith(linkPath))) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
    

});


// Check server status every 30 seconds
setInterval(checkServerStatus, 30000);

// Initial check on page load
window.addEventListener('load', checkServerStatus);