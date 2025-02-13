import logging
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)

class ClearSessionMiddleware(MiddlewareMixin):
    def process_response(self, request, response):
        # Vérifiez si l'utilisateur quitte la page spécifique
        if not (request.path.startswith('/workspace/') and '/datasets/' in request.path and '/model/get-custom-model/' in request.path):
            keys_to_clear = ['current_step', 'is_target','column_types' ,'data_target', 'model_type', 'model_name', 'preprocessing_steps', 'hyperparameters']
            for key in keys_to_clear:
                if key in request.session:
                    logger.debug(f"Suppression de la clé de session : {key}")
                    request.session.pop(key, None)
            request.session.save()
            logger.debug("Session nettoyée")
        return response